"""Train WMH2017 2.5D ConvNeXt-Tiny segmentation (active port)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.data.wmh_slice_dataset import (
    WmhSliceDataset,
    WmhVolumeDataset,
    resolve_modality_keys,
)
from wmh2017.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from wmh2017.training.losses import DiceFocalLoss, TverskyFocalLoss
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


# --- T1-R3: checkpoint persistence + validation-only probability diagnostics ---
# All additive; the selected-checkpoint contract (filename/state/alias/selection) is unchanged.

_VAL_PROB_THRESHOLDS = (0.15, 0.40, 0.50)


def new_val_prob_accum() -> dict[str, Any]:
    """Fresh streaming accumulator for per-epoch validation probability diagnostics."""
    return {"max_prob": 0.0, "counts": {t: 0 for t in _VAL_PROB_THRESHOLDS}, "lesion_probs": []}


def update_val_prob_accum(accum: dict[str, Any], prob: Any, label: Any) -> None:
    """Fold one validation batch's foreground prob + binary label into ``accum`` (streaming).

    Keeps only scalar accumulators and the (sparse) GT-lesion-voxel probabilities — never the full
    probability volume — so memory stays bounded. ``prob``/``label`` are array-likes of equal size.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    if p.size:
        accum["max_prob"] = max(float(accum["max_prob"]), float(p.max()))
    for t in accum["counts"]:
        accum["counts"][t] += int((p >= t).sum())
    lab = np.asarray(label).ravel()
    if lab.size == p.size and p.size:
        mask = lab > 0.5
        lp = p[mask]
        if lp.size:
            accum["lesion_probs"].append(lp.astype(np.float64))


def finalize_val_prob_diagnostics(accum: dict[str, Any]) -> dict[str, Any]:
    """Reduce a streaming accumulator to the per-epoch validation diagnostics dict.

    Empty validation / all-background labels yield 0.0 (no error).
    """
    lp = np.concatenate(accum["lesion_probs"]) if accum["lesion_probs"] else np.array([], dtype=np.float64)
    counts = accum["counts"]
    return {
        "val_max_prob": float(accum["max_prob"]),
        "val_pred_voxels_at_0_15": int(counts[0.15]),
        "val_pred_voxels_at_0_40": int(counts[0.40]),
        "val_pred_voxels_at_0_50": int(counts[0.50]),
        "val_lesion_voxel_mean_prob": float(lp.mean()) if lp.size else 0.0,
        "val_lesion_voxel_median_prob": float(np.median(lp)) if lp.size else 0.0,
    }


def checkpoint_filenames(selection_mode: str) -> dict[str, str | None]:
    """Checkpoint filenames for a mode. ``last`` is always present; ``loss_proxy_safety`` only in dice mode."""
    dice = selection_mode == "best_val_dice"
    return {
        "selected": "model_best_val_dice.pt" if dice else "model_best_val_loss_proxy.pt",
        "legacy_alias": "model_best.pt",
        "last": "model_last.pt",
        "loss_proxy_safety": "model_best_val_loss_proxy.pt" if dice else None,
    }


def build_checkpoint_inventory(
    selection_mode: str,
    *,
    best_epoch: int,
    best_score: float,
    best_val_loss: float,
    best_loss_epoch: int,
    last_epoch: int,
) -> dict[str, dict[str, Any]]:
    """Inventory of persisted checkpoints (filename -> role/epoch/metric) for run evidence."""
    dice = selection_mode == "best_val_dice"
    names = checkpoint_filenames(selection_mode)
    inv: dict[str, dict[str, Any]] = {}
    inv[str(names["selected"])] = {
        "role": "selected",
        "epoch": int(best_epoch),
        ("val_dice" if dice else "val_loss_proxy"): float(best_score),
    }
    inv[str(names["legacy_alias"])] = {"role": "selected_alias", "epoch": int(best_epoch)}
    if dice:
        inv["model_best_val_loss_proxy.pt"] = {
            "role": "safety_val_loss_proxy",
            "epoch": int(best_loss_epoch),
            "val_loss_proxy": float(best_val_loss),
        }
    inv[str(names["last"])] = {"role": "last", "epoch": int(last_epoch)}
    return inv


# --- T1-R5: default-off A+B retry enablers (loss dispatch + positive-slice balancing) ---
# Key-absent ⇒ bit-identical to the prior trainer (TverskyFocalLoss + shuffle DataLoader).

VALID_CONVNEXT_LOSSES = ("tversky_focal", "dice_focal")


def resolve_convnext_loss(train_cfg: dict[str, Any]) -> nn.Module:
    """Resolve the 2.5D training loss (default ``tversky_focal``; both 1-channel sigmoid).

    ``training.loss`` may be a str (name) or a dict with ``name`` + params. Absent/``tversky_focal``
    reproduces the historical `TverskyFocalLoss`; ``dice_focal`` returns the 1-channel `DiceFocalLoss`.
    Unknown names raise a clear ``ValueError``. MONAI 2-channel DiceCE is intentionally not used.
    """
    loss_cfg: Any = train_cfg.get("loss") or {}
    if isinstance(loss_cfg, str):
        name = loss_cfg.strip().lower()
        params: dict[str, Any] = {}
    elif isinstance(loss_cfg, dict):
        name = str(loss_cfg.get("name", "tversky_focal")).strip().lower()
        params = {k: v for k, v in loss_cfg.items() if k != "name"}
    else:
        raise ValueError(f"training.loss must be str or dict, got {type(loss_cfg)!r}")

    if name in {"tversky_focal", "tverskyfocal"}:
        return TverskyFocalLoss(
            alpha=float(params.get("alpha", 0.3)),
            beta=float(params.get("beta", 0.7)),
            gamma=float(params.get("gamma", 1.33)),
            smooth=float(params.get("smooth", 1.0)),
        )
    if name in {"dice_focal", "dicefocal"}:
        return DiceFocalLoss(
            alpha=float(params.get("alpha", 0.25)),
            gamma=float(params.get("gamma", 2.0)),
            smooth=float(params.get("smooth", 1.0)),
        )
    raise ValueError(f"unknown training.loss.name={name!r}; expected one of {list(VALID_CONVNEXT_LOSSES)}")


def compute_slice_foreground_flags(slice_ds: Any) -> list[bool]:
    """Per-slice foreground presence over ``slice_ds.index_map`` (one label load per volume).

    Duck-typed: needs ``slice_ds.index_map`` = list of ``(vidx, z)`` and
    ``slice_ds.volume_dataset[vidx]["label"]`` = a 3D mask. Does not modify the dataset.
    """
    flags: list[bool] = []
    cache_vidx: int | None = None
    cache_label: Any = None
    for vidx, z in slice_ds.index_map:
        if vidx != cache_vidx:
            cache_label = np.asarray(slice_ds.volume_dataset[int(vidx)]["label"])
            cache_vidx = vidx
        flags.append(bool(np.asarray(cache_label[int(z)]).any()))
    return flags


def positive_slice_sample_weights(flags: list[bool], pos_weight: float) -> list[float]:
    """Sampling weights: ``pos_weight`` for foreground slices, ``1.0`` for background (pure)."""
    pw = float(pos_weight)
    return [pw if f else 1.0 for f in flags]


def resolve_modalities(data_cfg: dict[str, Any]) -> list[str] | None:
    """Modality names from ``data.modalities`` (default-off: None ⇒ legacy single-modality)."""
    mods = data_cfg.get("modalities")
    if not mods:
        return None
    if isinstance(mods, str):
        mods = [mods]
    return [str(m).strip().lower() for m in mods]


def resolve_in_channels(model_cfg: dict[str, Any], k: int, n_modalities: int) -> int:
    """Model in_channels: explicit override else ``(2k+1)·n_modalities``."""
    return int(model_cfg.get("in_channels", (2 * int(k) + 1) * int(n_modalities)))


def build_train_loader(train_ds: Any, train_cfg: dict[str, Any]) -> DataLoader:
    """Train DataLoader. Default-off ⇒ existing ``shuffle=True`` loader (bit-identical).

    Opt-in via ``training.positive_slice_weight`` (>0): build a train-only
    ``WeightedRandomSampler`` that oversamples foreground-bearing slices. Validation is built
    separately and is unaffected.
    """
    batch_size = int(train_cfg.get("batch_size", 4))
    pos_weight = train_cfg.get("positive_slice_weight")
    if pos_weight is None:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    pw = float(pos_weight)
    if pw <= 0:
        raise ValueError(f"training.positive_slice_weight must be > 0; got {pw}")
    flags = compute_slice_foreground_flags(train_ds)
    weights = positive_slice_sample_weights(flags, pw)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)


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

    # T1-R8 default-off FLAIR+T1 parity: absent data.modalities ⇒ legacy single-modality (bit-identical).
    modalities = resolve_modalities(data_cfg)
    label_key = str(data_cfg.get("label_key", "wmh_path"))
    if modalities is not None and len(modalities) > 1:
        image_keys = resolve_modality_keys(modalities)  # validates names
        n_modalities = len(image_keys)
        vol_kwargs: dict[str, Any] = {"image_keys": image_keys, "label_key": label_key}
    else:
        image_key = (
            resolve_modality_keys(modalities)[0] if modalities else str(data_cfg.get("image_key", "flair_pre_path"))
        )
        n_modalities = 1
        vol_kwargs = {"image_key": image_key, "label_key": label_key}
    vol_ds = WmhVolumeDataset(data_cfg["dataset_manifest"], data_cfg["split_manifest"], "train", **vol_kwargs)
    val_vol_ds = WmhVolumeDataset(data_cfg["dataset_manifest"], data_cfg["split_manifest"], "val", **vol_kwargs)
    k = int(data_cfg.get("slice_k", 2))
    img_size_cfg = data_cfg.get("img_size")
    img_size = (int(img_size_cfg[0]), int(img_size_cfg[1])) if img_size_cfg else None
    train_ds = WmhSliceDataset(vol_ds, k=k, img_size=img_size)
    val_ds = WmhSliceDataset(val_vol_ds, k=k, img_size=img_size)
    train_loader = build_train_loader(train_ds, train_cfg)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    in_channels = resolve_in_channels(model_cfg, k, n_modalities)
    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        pretrained=bool(model_cfg.get("pretrained", True)),
        out_channels=1,
    ).to(device)
    loss_fn = resolve_convnext_loss(train_cfg)
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
    best_loss_epoch = -1
    last_epoch = -1
    selection_policy = dict(CONVNEXT_DICE_SELECTION_POLICY if dice_selection else CONVNEXT_SELECTION_POLICY)
    # Primary checkpoint name reveals selection semantics. Default (val_loss_proxy) keeps the
    # historical name; best_val_dice mode uses a Dice-best name. The legacy alias points to
    # whichever checkpoint the active mode selected.
    best_path = ckpt_dir / ("model_best_val_dice.pt" if dice_selection else "model_best_val_loss_proxy.pt")
    legacy_best_path = ckpt_dir / "model_best.pt"  # legacy alias only
    # T1-R3 (additive): always-persisted final checkpoint + dice-mode best-val-loss safety checkpoint.
    last_path = ckpt_dir / "model_last.pt"
    loss_proxy_safety_path = ckpt_dir / "model_best_val_loss_proxy.pt"

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

        last_epoch = epoch
        model.eval()
        val_losses: list[float] = []
        # Dice accumulators are only computed in best_val_dice mode (default path unchanged).
        dice_inter = 0.0
        dice_pred = 0.0
        dice_gt = 0.0
        # T1-R3 (additive): validation-only probability diagnostics, computed in every mode.
        prob_accum = new_val_prob_accum()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                val_losses.append(float(loss_fn(logits, labels).detach().cpu().item()))
                prob = torch.sigmoid(logits)
                update_val_prob_accum(prob_accum, prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                if dice_selection:
                    pred = (prob >= 0.5).float()
                    tgt = (labels > 0.5).float()
                    dice_inter += float((pred * tgt).sum().detach().cpu().item())
                    dice_pred += float(pred.sum().detach().cpu().item())
                    dice_gt += float(tgt.sum().detach().cpu().item())
        mean_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        # Track the min-val-loss epoch (same running-min value as before, now with the epoch index).
        loss_improved = mean_val < best_val_loss
        if loss_improved:
            best_val_loss = mean_val
            best_loss_epoch = epoch
        log_row: dict[str, Any] = {"epoch": epoch, "train_loss": mean_train, "val_loss": mean_val}
        if dice_selection:
            val_dice = micro_dice(dice_inter, dice_pred, dice_gt)
            log_row["val_dice"] = val_dice
            candidate = val_dice
        else:
            candidate = mean_val
        log_row.update(finalize_val_prob_diagnostics(prob_accum))
        logs.append(log_row)
        # T1-R3 (additive): in dice mode, also persist a best-val-loss safety checkpoint so a
        # loss-selected trained model is recoverable when val Dice is degenerate. (In val_loss_proxy
        # mode the selection block below already writes this exact file — gate avoids double-write.)
        if dice_selection and loss_improved:
            torch.save(  # nosec B614
                {
                    "run_id": run_id,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "selection_policy": dict(CONVNEXT_SELECTION_POLICY),
                    "best_val_loss_proxy": best_val_loss,
                    "best_selection_epoch": best_loss_epoch,
                    "checkpoint_kind": "safety_val_loss_proxy",
                    "claim_boundary": "local PoC ConvNeXt 2.5D training only",
                },
                loss_proxy_safety_path,
            )
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

    # T1-R3 (additive): always persist the final-epoch ("last") checkpoint, every mode, so a trained
    # model is recoverable even when the selection metric is degenerate.
    torch.save(  # nosec B614
        {
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "selection_policy": dict(selection_policy),
            "best_val_loss_proxy": best_val_loss,
            "checkpoint_kind": "last",
            "epoch": last_epoch,
            "claim_boundary": "local PoC ConvNeXt 2.5D training only",
        },
        last_path,
    )
    checkpoint_inventory = build_checkpoint_inventory(
        selection_mode,
        best_epoch=best_epoch,
        best_score=best_score,
        best_val_loss=best_val_loss,
        best_loss_epoch=best_loss_epoch,
        last_epoch=last_epoch,
    )

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
        "checkpoints": checkpoint_inventory,
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
