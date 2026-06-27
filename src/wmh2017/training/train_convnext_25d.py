"""Train WMH2017 2.5D ConvNeXt-Tiny segmentation (active port)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.data.wmh_slice_dataset import WmhSliceDataset, WmhVolumeDataset
from wmh2017.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from wmh2017.training.losses import TverskyFocalLoss
from wmh2017.training.mps_compat import enable_mps_cpu_fallback, resolve_training_device

# ConvNeXt 2.5D selects its best checkpoint by validation loss proxy (minimization).
# This is explicitly NOT Dice-best and NOT lesion-recall-best.
CONVNEXT_SELECTION_POLICY: dict[str, Any] = {
    "selection_metric": "val_loss_proxy",
    "selection_mode": "min",
    "checkpoint_semantics": "best validation loss proxy; not best Dice and not best lesion recall",
}

# Opt-in best-validation-Dice selection (default-off): makes the 2.5D ConvNeXt pilot
# comparable to the 3D best-on-val baseline. Key absent ⇒ unchanged val_loss_proxy.
CONVNEXT_DICE_SELECTION_POLICY: dict[str, Any] = {
    "selection_metric": "val_dice",
    "selection_mode": "max",
    "checkpoint_semantics": ("best validation Dice (micro over val slices @ threshold 0.5); not lesion-recall best"),
}

VALID_CHECKPOINT_SELECTION = ("val_loss_proxy", "best_val_dice")


def resolve_checkpoint_selection(train_cfg: dict[str, Any]) -> str:
    """Resolve the checkpoint-selection mode (default ``val_loss_proxy``).

    Validates ``training.checkpoint_selection`` against the supported modes and raises a
    clear ``ValueError`` for an unknown mode. Absent key ⇒ existing ``val_loss_proxy``.
    """
    mode = str(train_cfg.get("checkpoint_selection", "val_loss_proxy")).strip().lower()
    if mode not in VALID_CHECKPOINT_SELECTION:
        raise ValueError(
            f"unknown training.checkpoint_selection={mode!r}; " f"expected one of {list(VALID_CHECKPOINT_SELECTION)}"
        )
    return mode


def micro_dice(inter: float, pred_sum: float, gt_sum: float, eps: float = 1e-8) -> float:
    """Micro Dice ``2*inter / (pred_sum + gt_sum)`` accumulated over validation slices.

    Returns 1.0 when both prediction and ground truth are empty (matches the binary-Dice
    empty-score convention) so all-empty validation does not penalize selection.
    """
    denom = float(pred_sum) + float(gt_sum)
    if denom <= 0.0:
        return 1.0
    return float(2.0 * float(inter) / (denom + eps))


def is_better_selection(mode: str, candidate: float, best: float) -> bool:
    """Whether ``candidate`` strictly improves on ``best`` for the given mode.

    ``best_val_dice`` maximizes (candidate > best); ``val_loss_proxy`` minimizes
    (candidate < best). Strictly-better comparison means ties keep the earliest epoch,
    making selection deterministic.
    """
    if mode == "best_val_dice":
        return candidate > best
    return candidate < best


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_convnext_25d(config_path: str | Path) -> dict[str, Any]:
    enable_mps_cpu_fallback()
    cfg = _load_config(config_path)
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    run_id = str(run_cfg.get("run_id", "wmh2017_convnext25d"))
    seed = int(run_cfg.get("seed", 42))
    _set_seed(seed)
    device, device_runtime = resolve_training_device(torch, str(run_cfg.get("device", "auto")))

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    vol_ds = WmhVolumeDataset(
        data_cfg["dataset_manifest"],
        data_cfg["split_manifest"],
        "train",
        image_key=str(data_cfg.get("image_key", "flair_pre_path")),
        label_key=str(data_cfg.get("label_key", "wmh_path")),
    )
    val_vol_ds = WmhVolumeDataset(
        data_cfg["dataset_manifest"],
        data_cfg["split_manifest"],
        "val",
        image_key=str(data_cfg.get("image_key", "flair_pre_path")),
        label_key=str(data_cfg.get("label_key", "wmh_path")),
    )
    k = int(data_cfg.get("slice_k", 2))
    img_size_cfg = data_cfg.get("img_size")
    img_size = (int(img_size_cfg[0]), int(img_size_cfg[1])) if img_size_cfg else None
    train_ds = WmhSliceDataset(vol_ds, k=k, img_size=img_size)
    val_ds = WmhSliceDataset(val_vol_ds, k=k, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 4)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    in_channels = int(model_cfg.get("in_channels", 2 * k + 1))
    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        pretrained=bool(model_cfg.get("pretrained", True)),
        out_channels=1,
    ).to(device)
    loss_fn = TverskyFocalLoss(
        alpha=float(train_cfg.get("loss", {}).get("alpha", 0.3)),
        beta=float(train_cfg.get("loss", {}).get("beta", 0.7)),
        gamma=float(train_cfg.get("loss", {}).get("gamma", 1.33)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))

    max_epochs = int(train_cfg.get("max_epochs", 20))
    selection_mode = resolve_checkpoint_selection(train_cfg)
    dice_selection = selection_mode == "best_val_dice"
    logs: list[dict[str, Any]] = []
    # best_score tracks the active selection metric (dice: maximize; loss: minimize);
    # best_val_loss is always tracked separately for evidence/transparency.
    best_score = float("-inf") if dice_selection else float("inf")
    best_val_loss = float("inf")
    best_val_dice = float("-inf")
    best_epoch = -1
    selection_policy = dict(CONVNEXT_DICE_SELECTION_POLICY if dice_selection else CONVNEXT_SELECTION_POLICY)
    # Primary checkpoint name reveals selection semantics. Default (val_loss_proxy) keeps the
    # historical name; best_val_dice mode uses a Dice-best name. The legacy alias points to
    # whichever checkpoint the active mode selected.
    best_path = ckpt_dir / ("model_best_val_dice.pt" if dice_selection else "model_best_val_loss_proxy.pt")
    legacy_best_path = ckpt_dir / "model_best.pt"  # legacy alias only

    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            if isinstance(logits, list):
                logits = logits[0]
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}")
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        # Dice accumulators are only computed in best_val_dice mode (default path unchanged).
        dice_inter = 0.0
        dice_pred = 0.0
        dice_gt = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                val_losses.append(float(loss_fn(logits, labels).detach().cpu().item()))
                if dice_selection:
                    pred = (torch.sigmoid(logits) >= 0.5).float()
                    tgt = (labels > 0.5).float()
                    dice_inter += float((pred * tgt).sum().detach().cpu().item())
                    dice_pred += float(pred.sum().detach().cpu().item())
                    dice_gt += float(tgt.sum().detach().cpu().item())
        mean_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        best_val_loss = min(best_val_loss, mean_val)
        log_row: dict[str, Any] = {"epoch": epoch, "train_loss": mean_train, "val_loss": mean_val}
        if dice_selection:
            val_dice = micro_dice(dice_inter, dice_pred, dice_gt)
            log_row["val_dice"] = val_dice
            candidate = val_dice
        else:
            candidate = mean_val
        logs.append(log_row)
        if is_better_selection(selection_mode, candidate, best_score):
            best_score = candidate
            best_epoch = epoch
            if dice_selection:
                best_val_dice = candidate
            payload = {
                "run_id": run_id,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "selection_policy": dict(selection_policy),
                "best_val_loss_proxy": best_val_loss,
                "best_selection_score": best_score,
                "best_selection_epoch": best_epoch,
                "claim_boundary": "local PoC ConvNeXt 2.5D training only",
            }
            if dice_selection:
                payload["best_val_dice"] = best_val_dice
            torch.save(payload, best_path)  # nosec B614
            torch.save(payload, legacy_best_path)  # nosec B614 — legacy alias copy

    log_path = log_dir / "train_log.jsonl"
    log_path.write_text("\n".join(json.dumps(x) for x in logs) + "\n", encoding="utf-8")
    evidence = {
        "run_id": run_id,
        "status": "completed",
        "training_mode": "convnext_25d",
        "device": str(device),
        **device_runtime,
        "checkpoint_path": str(best_path),
        "legacy_checkpoint_alias": str(legacy_best_path),
        "train_log": str(log_path),
        "selection_policy": dict(selection_policy),
        "best_val_loss_proxy": best_val_loss,
        "best_selection_score": best_score,
        "best_selection_epoch": best_epoch,
        "metric_limitations": (
            [
                "This checkpoint is selected by validation Dice (micro @0.5), not lesion recall.",
                f"model_best.pt is a legacy alias of {best_path.name}.",
            ]
            if dice_selection
            else [
                "This checkpoint is not selected by mean Dice.",
                "This checkpoint is not selected by lesion recall.",
                "model_best.pt is a legacy alias of model_best_val_loss_proxy.pt.",
            ]
        ),
        "safety": {
            "test_split_used": False,
            "label_policy": "label==1 foreground; label==2 ignored as foreground",
            "claim_boundary": "local PoC ConvNeXt 2.5D training only",
        },
    }
    if dice_selection:
        evidence["best_val_dice"] = best_val_dice
    write_json(out_dir / "run_evidence.json", evidence)
    notes = (
        f"convnext_25d; best_val_dice={best_val_dice:.6f}"
        if dice_selection
        else f"convnext_25d; best_val_loss_proxy={best_val_loss:.6f}"
    )
    append_run_manifest(
        make_run_row(
            run_id=run_id,
            run_purpose="wmh2017_convnext_25d_training",
            config_path=str(config_path),
            dataset_manifest=str(data_cfg["dataset_manifest"]),
            split_manifest=str(data_cfg["split_manifest"]),
            model_name="convnext_nnunet_25d",
            model_version="full",
            seed=seed,
            device=str(device),
            status="completed",
            checkpoint_path=str(best_path),
            prediction_dir="",
            notes=notes,
        ),
        run_cfg.get("run_manifest", "registry/runs/run_manifest.csv"),
    )
    result = {"run_id": run_id, "checkpoint_path": str(best_path), "best_val_loss_proxy": best_val_loss}
    if dice_selection:
        result["best_val_dice"] = best_val_dice
    return result
