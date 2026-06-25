"""Executable MONAI smoke/full training for WMH2017.

The goal is not SOTA performance. The goal is to prove that data loading,
label==1 policy, train/val split, one short training loop, validation inference,
prediction persistence and run evidence work end-to-end.

Safety boundaries:
- challenge_split=test is never consumed by this module.
- label value 2 is ignored as foreground by converting labels to label==1.
- raw data and checkpoints remain outside git-controlled source paths by default.
"""

from __future__ import annotations

import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from wmh2017.audit.run_labeling import (
    architecture_parity_block,
    checkpoint_filename,
    completion_message,
    mps_execution_claim,
    run_purpose_for_mode,
)
from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.config.training_config import (
    modalities_to_payload,
    modality_keys,
    resolve_input_modalities,
)
from wmh2017.data.case_records import case_records_to_monai_rows, load_case_records
from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_wmh_label1
from wmh2017.evaluation.voxel_metrics import dice_wmh_label1
from wmh2017.inference.export_probabilities import infer_foreground_probability, save_case_prediction
from wmh2017.inference.input_builder import load_normalized_input_volume, to_batched_tensor
from wmh2017.io.images import load_array
from wmh2017.models.factory import build_unet
from wmh2017.training.loss_factory import build_loss
from wmh2017.training.mps_compat import (
    apply_mps_safe_convtranspose_patch,
    enable_mps_cpu_fallback,
    record_mps_convtranspose_patch,
    resolve_training_device,
)
from wmh2017.training.selection import (
    evaluate_candidate,
    policy_to_payload,
    selection_policy_from_config,
)
from wmh2017.training.transforms import build_monai_transforms


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for YAML configs. Install requirements-lock.txt.") from e
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _require_monai_stack() -> tuple[Any, Any]:
    try:
        import torch
        from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
        from monai.inferers import sliding_window_inference
        from monai.losses import DiceCELoss
        from monai.networks.nets import UNet
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            Lambdad,
            LoadImaged,
            NormalizeIntensityd,
            RandAffined,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandShiftIntensityd,
            ResizeWithPadOrCropd,
        )
    except ImportError as e:
        raise ImportError(
            "MONAI smoke training requires torch, monai and nibabel. "
            "Install with: python -m pip install -r requirements-lock.txt"
        ) from e
    return torch, {
        "CacheDataset": CacheDataset,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "decollate_batch": decollate_batch,
        "sliding_window_inference": sliding_window_inference,
        "DiceCELoss": DiceCELoss,
        "UNet": UNet,
        "Compose": Compose,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "EnsureTyped": EnsureTyped,
        "Lambdad": Lambdad,
        "LoadImaged": LoadImaged,
        "NormalizeIntensityd": NormalizeIntensityd,
        "RandCropByPosNegLabeld": RandCropByPosNegLabeld,
        "RandFlipd": RandFlipd,
        "RandAffined": RandAffined,
        "RandShiftIntensityd": RandShiftIntensityd,
        "ResizeWithPadOrCropd": ResizeWithPadOrCropd,
    }


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _case_rows(
    manifest_csv: str | Path, split_csv: str | Path, assigned_split: str, image_key: str, label_key: str
) -> list[dict[str, str]]:
    """Resolve legacy single-modality MONAI rows.

    Thin wrapper over :func:`load_case_records` / :func:`case_records_to_monai_rows`.
    Kept for backward-compatible imports; returns ``{"case_id", "image", "label"}`` rows.
    """
    input_modalities = resolve_input_modalities({"image_key": image_key})
    records = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split=assigned_split,
        input_modalities=input_modalities,
        label_key=label_key,
    )
    return case_records_to_monai_rows(records)


def _build_model(monai: dict[str, Any], cfg: dict[str, Any]) -> Any:
    """Build the MONAI UNet (delegates to :func:`models.factory.build_unet`)."""
    return build_unet(monai, cfg)


def _transforms(
    monai: dict[str, Any],
    patch_size: list[int] | tuple[int, int, int],
    train: bool,
    train_cfg: dict[str, Any] | None = None,
    input_keys: tuple[str, ...] = ("image",),
) -> Any:
    return build_monai_transforms(monai, patch_size, train=train, train_cfg=train_cfg, input_keys=input_keys)


def _training_mode(train_cfg: dict[str, Any]) -> str:
    return str(train_cfg.get("mode", "smoke")).lower()


def _peak_rss_kb(raw_rss: int) -> float:
    """Normalize ru_maxrss to kilobytes across Linux (KB) and macOS (bytes)."""
    if sys.platform == "darwin":
        return round(float(raw_rss) / 1024.0, 3)
    return round(float(raw_rss), 3)


def _amp_policy(use_amp: bool, device_type: str) -> tuple[bool, str | None, str]:
    """Return AMP/autocast settings with an explicit compute-precision policy.

    Speed on Apple Silicon comes from the MPS device, not fp16 autocast.
    MPS autocast is disabled to preserve float32 numerical fidelity in this PoC.
    """
    if use_amp and device_type == "cuda":
        return True, "cuda", "cuda_amp_optional"
    if device_type == "mps":
        return False, None, "mps_float32_accuracy_first"
    return False, None, "float32_full_precision"


def build_lr_scheduler(
    torch: Any,
    optimizer: Any,
    train_cfg: dict[str, Any],
    *,
    max_epochs: int,
) -> tuple[Any, dict[str, Any]]:
    """Build a config-gated per-epoch LR scheduler (backward compatible).

    If ``training.lr_scheduler`` is absent or name in {none, "", constant, fixed},
    returns (None, {...enabled: False}) and the run keeps the prior fixed LR.
    Supported: cosine (CosineAnnealingLR), poly (polynomial decay), step (StepLR).
    """
    sched_cfg = train_cfg.get("lr_scheduler") or {}
    name = str(sched_cfg.get("name", "none")).lower()
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if warmup_epochs < 0 or warmup_epochs >= max_epochs:
        raise ValueError(f"warmup_epochs must be in [0, max_epochs); got {warmup_epochs}")
    if name in {"none", "", "constant", "fixed"}:
        return None, {"enabled": False, "name": "constant"}

    if name == "cosine":
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        t_max = int(sched_cfg.get("t_max", max_epochs))
        if warmup_epochs > 0:
            # Short linear warmup (lr ramps from peak/warmup_epochs to peak over warmup_epochs),
            # then cosine-anneal over the remaining epochs. Per-epoch stepping; works with any
            # number of optimizer param groups.
            warm = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / warmup_epochs, end_factor=1.0, total_iters=warmup_epochs
            )
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, t_max - warmup_epochs), eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm, main], milestones=[warmup_epochs]
            )
            return scheduler, {
                "enabled": True,
                "name": "cosine_warmup",
                "t_max": t_max,
                "eta_min": eta_min,
                "warmup_epochs": warmup_epochs,
            }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return scheduler, {"enabled": True, "name": "cosine", "t_max": t_max, "eta_min": eta_min}

    if name == "poly":
        power = float(sched_cfg.get("power", 0.9))
        total = max(1, int(sched_cfg.get("max_epochs", max_epochs)))

        def _poly(epoch: int) -> float:
            return (1.0 - min(epoch, total) / total) ** power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly)
        return scheduler, {"enabled": True, "name": "poly", "power": power, "max_epochs": total}

    if name == "step":
        step_size = int(sched_cfg.get("step_size", max(1, max_epochs // 3)))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, {"enabled": True, "name": "step", "step_size": step_size, "gamma": gamma}

    raise ValueError(f"unsupported lr_scheduler name: {name}")


def _build_train_dataset(
    monai: dict[str, Any],
    train_rows: list[dict[str, str]],
    patch_size: list[int],
    cache_rate: float,
    train_cfg: dict[str, Any] | None = None,
    input_keys: tuple[str, ...] = ("image",),
) -> Any:
    transform = _transforms(monai, patch_size, train=True, train_cfg=train_cfg, input_keys=input_keys)
    if cache_rate > 0:
        return monai["CacheDataset"](data=train_rows, transform=transform, cache_rate=float(cache_rate))
    return monai["Dataset"](data=train_rows, transform=transform)


def _run_validation_metrics(
    *,
    model: Any,
    torch: Any,
    monai: dict[str, Any],
    val_rows: list[dict[str, str]],
    patch_size: list[int],
    device: Any,
    threshold: float,
    input_keys: tuple[str, ...] = ("image",),
) -> tuple[dict[str, float], int]:
    """Run validation inference and return mean dice/lesion_recall/lesion_f1.

    Lesion metrics are computed so that selection_metric can be dice, recall, f1, or
    a composite. This is local validation only; not a SOTA/official/clinical claim.
    """
    if not val_rows:
        return {"mean_dice": 0.0, "mean_lesion_recall": 0.0, "mean_lesion_f1": 0.0}, 0
    model.eval()
    roi_size = tuple(patch_size)
    dice_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    with torch.no_grad():
        for row in val_rows:
            label = load_array(row["label"])
            label_mask = (label == 1).astype(np.uint8)
            image_paths = {key: row[key] for key in input_keys}
            volume = load_normalized_input_volume(image_paths=image_paths, input_keys=input_keys)
            tensor = to_batched_tensor(torch, volume, device)
            logits = monai["sliding_window_inference"](tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs[0].detach().cpu().numpy() >= threshold).astype(np.uint8)
            dice_scores.append(float(dice_wmh_label1(pred, label_mask)))
            lesion = lesion_recall_f1_wmh_label1(pred, label_mask)
            recall_scores.append(float(lesion["lesion_recall"]))
            f1_scores.append(float(lesion["lesion_f1"]))
    metrics = {
        "mean_dice": float(np.mean(dice_scores)),
        "mean_lesion_recall": float(np.mean(recall_scores)),
        "mean_lesion_f1": float(np.mean(f1_scores)),
    }
    return metrics, len(dice_scores)


def _save_predictions(
    *,
    model: Any,
    torch: Any,
    monai: dict[str, Any],
    val_rows: list[dict[str, str]],
    patch_size: list[int],
    device: Any,
    threshold: float,
    pred_dir: Path,
    input_keys: tuple[str, ...] = ("image",),
) -> int:
    probs_dir = pred_dir / "probs"
    count = 0
    for row in val_rows:
        image_paths = {key: row[key] for key in input_keys}
        reference_image_path = row[input_keys[0]]
        probs = infer_foreground_probability(
            model=model,
            torch=torch,
            monai=monai,
            image_paths=image_paths,
            input_keys=input_keys,
            patch_size=patch_size,
            device=device,
        )
        save_case_prediction(
            probs=probs,
            threshold=threshold,
            reference_image_path=reference_image_path,
            pred_path=pred_dir / f"{row['case_id']}_pred.nii.gz",
            prob_path=probs_dir / f"{row['case_id']}.npz",
        )
        count += 1
    return count


def require_train_val_rows(
    train_rows: list[Any],
    val_rows: list[Any],
    *,
    allow_empty_val: bool,
    save_last_checkpoint: bool,
) -> None:
    """Validate train/val row presence against the all-train flags.

    Default behavior (allow_empty_val=False) is unchanged: train and val are both required.
    All-train mode (allow_empty_val=True) requires save_last_checkpoint=True and an EMPTY val set —
    the final model is the last-epoch checkpoint with no validation- or test-based selection.
    """
    if not train_rows:
        raise ValueError("train rows are required for training")
    if allow_empty_val:
        if not save_last_checkpoint:
            raise ValueError("allow_empty_val=true requires save_last_checkpoint=true")
        if val_rows:
            raise ValueError("allow_empty_val=true requires an empty val set (all-train mode)")
        return
    if not val_rows:
        raise ValueError("train and val rows are required for training")


def resolve_checkpoint_policy(*, run_val: bool, save_last_checkpoint: bool) -> str:
    """Return the checkpoint-selection policy label recorded in run evidence."""
    return "last_epoch" if (not run_val and save_last_checkpoint) else "best_on_val"


def main(config_path: str) -> None:
    enable_mps_cpu_fallback()
    torch, monai = _require_monai_stack()
    cfg = _load_config(config_path)

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    run_id = str(run_cfg.get("run_id", "wmh2017_monai_smoke"))
    seed = int(run_cfg.get("seed", 42))
    _set_seed(seed, torch)
    device_requested = str(run_cfg.get("device", "auto"))
    device, device_runtime = resolve_training_device(torch, device_requested)

    dataset_manifest = str(data_cfg["dataset_manifest"])
    split_manifest = str(data_cfg["split_manifest"])
    input_modalities = resolve_input_modalities(data_cfg)
    input_keys = modality_keys(input_modalities)
    label_key = str(data_cfg.get("label_key", "wmh_path"))
    patch_size = list(data_cfg.get("patch_size", [32, 32, 32]))
    num_workers_requested = int(data_cfg.get("num_workers", 0))
    num_workers = num_workers_requested
    if sys.platform == "darwin" and num_workers > 0:
        num_workers = 0
    cache_rate = float(data_cfg.get("cache_rate", 0.0))
    val_max_cases = int(data_cfg.get("val_max_cases", 2))
    mode = _training_mode(train_cfg)
    threshold = float(train_cfg.get("threshold", 0.5))
    use_amp = bool(train_cfg.get("use_amp", False)) and mode == "full"

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    for p in [pred_dir, ckpt_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    train_rows = case_records_to_monai_rows(
        load_case_records(
            manifest_csv=dataset_manifest,
            split_csv=split_manifest,
            assigned_split="train",
            input_modalities=input_modalities,
            label_key=label_key,
        )
    )
    all_val_rows = case_records_to_monai_rows(
        load_case_records(
            manifest_csv=dataset_manifest,
            split_csv=split_manifest,
            assigned_split="val",
            input_modalities=input_modalities,
            label_key=label_key,
        )
    )
    if val_max_cases > 0:
        val_rows = all_val_rows[:val_max_cases]
        val_eval_rows = all_val_rows[:val_max_cases]
    else:
        val_rows = all_val_rows
        val_eval_rows = all_val_rows
    allow_empty_val = bool(train_cfg.get("allow_empty_val", False))
    save_last_checkpoint = bool(train_cfg.get("save_last_checkpoint", False))
    require_train_val_rows(
        train_rows,
        val_rows,
        allow_empty_val=allow_empty_val,
        save_last_checkpoint=save_last_checkpoint,
    )
    run_val = bool(val_eval_rows)

    train_ds = _build_train_dataset(monai, train_rows, patch_size, cache_rate, train_cfg, input_keys=input_keys)
    train_loader = monai["DataLoader"](train_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    model = build_unet(monai, cfg, input_modalities=input_modalities)
    if device.type == "mps":
        patched_layers = apply_mps_safe_convtranspose_patch(model)
        if patched_layers == 0:
            raise RuntimeError(
                "MPS was selected but no ConvTranspose3d layers were replaced; "
                "MONAI smoke cannot run on MPS without the compatibility patch."
            )
        device_runtime = record_mps_convtranspose_patch(device_runtime, patched_layers)
    model = model.to(device)
    loss_fn = build_loss(train_cfg, monai)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))
    lr_scheduler, lr_scheduler_info = build_lr_scheduler(
        torch, opt, train_cfg, max_epochs=int(train_cfg.get("max_epochs", 1))
    )

    max_epochs = int(train_cfg.get("max_epochs", 1))
    max_steps = int(train_cfg.get("max_steps_per_epoch", 2))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
    selection_policy = selection_policy_from_config(train_cfg, default_metric="mean_dice")
    checkpoint_semantics = (
        f"best local validation {selection_policy.metric} ({selection_policy.mode}); "
        "not test split, not SOTA, not clinical, not production"
    )
    selection_policy_payload = policy_to_payload(selection_policy, checkpoint_semantics=checkpoint_semantics)
    logs: list[dict[str, Any]] = []
    best_score: float | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0
    checkpoint_path = ""
    best_checkpoint_path = str(ckpt_dir / checkpoint_filename(mode))
    best_alias_path = str(ckpt_dir / f"model_best_{selection_policy.metric}.pt")
    last_checkpoint_path = str(ckpt_dir / "model_last.pt")
    last_checkpoint_saved = False
    amp_enabled, autocast_device, amp_precision_policy = _amp_policy(use_amp, device.type)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    first_epoch_resource: dict[str, Any] | None = None

    for epoch in range(max_epochs):
        epoch_started = time.perf_counter()
        epoch_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            if mode == "smoke" and step >= max_steps:
                break
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)
            opt.zero_grad(set_to_none=True)
            if amp_enabled and autocast_device:
                with torch.autocast(device_type=autocast_device, enabled=True):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}: {loss.item()}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            logs.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": loss_value,
                    "mode": mode,
                }
            )

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if run_val:
            val_metrics, val_n = _run_validation_metrics(
                model=model,
                torch=torch,
                monai=monai,
                val_rows=val_eval_rows,
                patch_size=patch_size,
                device=device,
                threshold=threshold,
                input_keys=input_keys,
            )
        else:
            # All-train mode: no validation set, no per-epoch metrics, no selection.
            val_metrics = {"mean_dice": 0.0, "mean_lesion_recall": 0.0, "mean_lesion_f1": 0.0}
            val_n = 0
        val_dice = val_metrics["mean_dice"]
        if mode == "full" and epoch == 0:
            epoch_rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            first_epoch_resource = {
                "epoch_index": 0,
                "wall_time_seconds": round(time.perf_counter() - epoch_started, 3),
                "peak_rss_kb": _peak_rss_kb(max(epoch_rss_before, epoch_rss_after)),
                "train_case_count": len(train_rows),
                "val_case_count": val_n,
                "patch_size": patch_size,
                "max_epochs_configured": max_epochs,
            }
        if run_val:
            logs.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_dice": val_dice,
                    "val_mean_lesion_recall": val_metrics["mean_lesion_recall"],
                    "val_mean_lesion_f1": val_metrics["mean_lesion_f1"],
                    "val_n_cases": val_n,
                    "mean_epoch_loss": mean_epoch_loss,
                    "selection_metric": selection_policy.metric,
                    "mode": mode,
                }
            )
            decision = evaluate_candidate(
                selection_policy,
                val_metrics,
                best_score=best_score,
                best_metrics=best_metrics or None,
            )
            if decision.improved:
                best_score = decision.score
                best_metrics = dict(val_metrics)
                best_epoch = epoch
                epochs_without_improvement = 0
                if mode == "full" and bool(train_cfg.get("save_best_only", True)):
                    payload = {
                        "run_id": run_id,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "input_modalities": modalities_to_payload(input_modalities),
                        "global_step": global_step,
                        "selection_policy": selection_policy_payload,
                        "best_selection_score": best_score,
                        "best_selection_epoch": best_epoch,
                        "best_metrics": best_metrics,
                        "best_val_dice": best_metrics["mean_dice"],  # legacy alias
                        "best_epoch": best_epoch,
                        "checkpoint_semantics": checkpoint_semantics,
                        "claim_boundary": "local PoC full training; not clinical or production model",
                    }
                    torch.save(  # nosec B614 — local trusted best checkpoint only; not loading untrusted weights
                        payload,
                        best_checkpoint_path,
                    )
                    # Metric-explicit alias so the filename reveals selection semantics.
                    if best_alias_path != best_checkpoint_path:
                        torch.save(payload, best_alias_path)  # nosec B614 — local trusted copy
                    checkpoint_path = best_checkpoint_path
            else:
                epochs_without_improvement += 1
        else:
            logs.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_skipped": True,
                    "val_n_cases": 0,
                    "mean_epoch_loss": mean_epoch_loss,
                    "selection_metric": selection_policy.metric,
                    "mode": mode,
                }
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        if run_val and mode == "full" and epochs_without_improvement >= early_stopping_patience:
            break

    if mode == "full" and save_last_checkpoint:
        last_payload = {
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "input_modalities": modalities_to_payload(input_modalities),
            "global_step": global_step,
            "checkpoint_semantics": "final/last training epoch; no validation- or test-based selection",
            "claim_boundary": "local PoC full training; not clinical or production model",
        }
        torch.save(last_payload, last_checkpoint_path)  # nosec B614 — local trusted last checkpoint
        last_checkpoint_saved = True
        # In all-train (empty-val) mode no best checkpoint exists, so the last epoch is selected.
        if not checkpoint_path:
            checkpoint_path = last_checkpoint_path

    if mode == "smoke" and bool(train_cfg.get("save_checkpoint", True)):
        checkpoint_path = str(ckpt_dir / checkpoint_filename(mode))
        torch.save(  # nosec B614 — local smoke checkpoint only; not loading untrusted weights
            {
                "run_id": run_id,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "input_modalities": modalities_to_payload(input_modalities),
                "global_step": global_step,
                "claim_boundary": "smoke checkpoint only; not a clinical or production model",
            },
            checkpoint_path,
        )

    if bool(train_cfg.get("save_predictions", True)):
        if mode == "full" and checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(  # nosec B614 — reload local best checkpoint written by this run only
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(state["model_state_dict"])
        pred_count = _save_predictions(
            model=model,
            torch=torch,
            monai=monai,
            val_rows=val_rows,
            patch_size=patch_size,
            device=device,
            threshold=threshold,
            pred_dir=pred_dir,
            input_keys=input_keys,
        )
    else:
        pred_count = 0

    log_path = log_dir / "train_log.jsonl"
    log_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in logs) + "\n", encoding="utf-8")

    evidence = {
        "run_id": run_id,
        "status": "completed",
        "training_mode": mode,
        "device": str(device),
        **device_runtime,
        "global_step": global_step,
        "selection_policy": selection_policy_payload,
        "checkpoint_policy": (
            resolve_checkpoint_policy(run_val=run_val, save_last_checkpoint=save_last_checkpoint)
            if mode == "full"
            else "smoke"
        ),
        "allow_empty_val": allow_empty_val,
        "save_last_checkpoint": save_last_checkpoint,
        "val_case_count": len(val_eval_rows),
        "last_checkpoint_path": last_checkpoint_path if last_checkpoint_saved else "",
        "best_selection_score": best_score if mode == "full" else None,
        "best_selection_epoch": best_epoch if mode == "full" else None,
        "best_metrics": best_metrics if mode == "full" else None,
        "legacy_best_val_dice": best_metrics.get("mean_dice") if mode == "full" else None,
        "best_val_dice": best_metrics.get("mean_dice") if mode == "full" else None,  # legacy alias
        "best_epoch": best_epoch if mode == "full" else None,
        "selection_claim_note": (
            "best_* fields describe the selected local-validation snapshot only; "
            "smoke mode emits null best_* (non-performance-claim)"
        ),
        "train_case_count": len(train_rows),
        "val_prediction_count": pred_count if bool(train_cfg.get("save_predictions", True)) else 0,
        "train_log": str(log_path),
        "checkpoint_path": checkpoint_path,
        "prediction_dir": str(pred_dir),
        "data_loader": {
            "num_workers_requested": num_workers_requested,
            "num_workers_effective": num_workers,
            "darwin_spawn_policy": "num_workers forced to 0 on macOS for MONAI DataLoader stability",
        },
        "amp": {
            "requested": use_amp,
            "effective": amp_enabled,
            "autocast_device": autocast_device,
            "compute_precision": "float32"
            if device.type in {"mps", "cpu"}
            else ("mixed" if amp_enabled else "float32"),
            "precision_policy": amp_precision_policy,
            "policy": (
                "Speed on MPS via GPU backend; accuracy preserved with float32 (no MPS fp16 autocast). "
                "CUDA may use AMP when use_amp=true."
            ),
        },
        "lr_scheduler": lr_scheduler_info,
        "resource": {"first_epoch": first_epoch_resource} if first_epoch_resource else None,
        "threshold_policy": {
            "training_threshold": threshold,
            "sweep_best_threshold": None,
            "sweep_split": "val",
        },
        "safety": {
            "test_split_used": False,
            "label_policy": "label==1 foreground; label==2 ignored as foreground",
            "claim_boundary": "local PoC smoke only" if mode == "smoke" else "local PoC full training only",
            "mps_execution_claim": mps_execution_claim(
                device.type,
                patched=bool(device_runtime.get("mps_convtranspose_patched", False)),
                mode=mode,
            ),
            "architecture_parity": architecture_parity_block(
                device_type=device.type,
                patched=bool(device_runtime.get("mps_convtranspose_patched", False)),
                patch_name=str(device_runtime.get("model_patch") or "ConvTranspose3d_to_InterpConv3d"),
            ),
        },
    }
    evidence_path = out_dir / "run_evidence.json"
    write_json(evidence_path, evidence)

    row = make_run_row(
        run_id=run_id,
        run_purpose=run_purpose_for_mode(mode),
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        model_name=str(cfg.get("model", {}).get("name", "monai_unet")),
        model_version="smoke" if mode == "smoke" else "full",
        seed=seed,
        device=str(device),
        status="completed",
        checkpoint_path=checkpoint_path,
        prediction_dir=str(pred_dir),
        notes=(
            f"mode={mode}; global_step={global_step}; "
            f"selection_metric={selection_policy.metric}; "
            f"best_selection_score={best_score if best_score is not None else float('nan'):.6f}; "
            f"best_val_dice={best_metrics.get('mean_dice', float('nan')):.6f}; no test110 use"
        ),
    )
    append_run_manifest(row, run_cfg.get("run_manifest", "registry/runs/run_manifest.csv"))

    print(completion_message(mode, run_id))
    print(f"Wrote train log: {log_path}")
    print(f"Wrote run evidence: {evidence_path}")
    if checkpoint_path:
        print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote predictions: {pred_dir}")
