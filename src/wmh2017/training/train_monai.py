"""Executable MONAI smoke training for WMH2017.

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
import pandas as pd

from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.data.preprocessing import normalize_nonzero_channelwise
from wmh2017.evaluation.voxel_metrics import dice_wmh_label1
from wmh2017.io.images import load_array, save_array_like
from wmh2017.training.mps_compat import (
    apply_mps_safe_convtranspose_patch,
    enable_mps_cpu_fallback,
    record_mps_convtranspose_patch,
    resolve_training_device,
)


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
            RandCropByPosNegLabeld,
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
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    if split.empty:
        raise ValueError(f"no rows for assigned_split={assigned_split} in {split_csv}")
    rows: list[dict[str, str]] = []
    for _, s in split.iterrows():
        case_id = str(s["case_id"])
        m = manifest[manifest["case_id"].astype(str) == case_id]
        if m.empty:
            raise ValueError(f"case_id={case_id} exists in split but not manifest")
        r = m.iloc[0]
        if str(r.get("challenge_split", "")).lower() == "test":
            raise ValueError(f"test case cannot be used for train/val smoke training: {case_id}")
        image = str(r.get(image_key, "") or r.get("flair_path", "") or r.get("flair_pre_path", ""))
        label = str(r.get(label_key, "") or r.get("wmh_path", "") or r.get("mask_path", ""))
        if not image or not label:
            raise ValueError(f"case_id={case_id} missing image or label path")
        rows.append({"case_id": case_id, "image": image, "label": label})
    return rows


def _build_model(monai: dict[str, Any], cfg: dict[str, Any]) -> Any:
    m = cfg.get("model", {})
    return monai["UNet"](
        spatial_dims=int(m.get("spatial_dims", 3)),
        in_channels=int(m.get("in_channels", 1)),
        out_channels=int(m.get("out_channels", 2)),
        channels=tuple(m.get("channels", [8, 16, 32])),
        strides=tuple(m.get("strides", [2, 2])),
        num_res_units=int(m.get("num_res_units", 1)),
    )


def _label_to_foreground_mask(label: np.ndarray) -> np.ndarray:
    """Map WMH label==1 foreground; label==2 ignored. Module-level for DataLoader pickling."""
    return (label == 1).astype(np.int64)


def _transforms(monai: dict[str, Any], patch_size: list[int] | tuple[int, int, int], train: bool) -> Any:
    ops = [
        monai["LoadImaged"](keys=["image", "label"]),
        monai["EnsureChannelFirstd"](keys=["image", "label"]),
        monai["Lambdad"](keys=["image"], func=normalize_nonzero_channelwise),
        monai["Lambdad"](keys=["label"], func=_label_to_foreground_mask),
    ]
    if train:
        ops.append(
            monai["RandCropByPosNegLabeld"](
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(patch_size),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            )
        )
    ops.extend(
        [
            monai["ResizeWithPadOrCropd"](keys=["image", "label"], spatial_size=tuple(patch_size)),
            monai["EnsureTyped"](keys=["image", "label"]),
        ]
    )
    return monai["Compose"](ops)


def _normalize_for_inference(image: np.ndarray) -> np.ndarray:
    return normalize_nonzero_channelwise(image)


def _training_mode(train_cfg: dict[str, Any]) -> str:
    return str(train_cfg.get("mode", "smoke")).lower()


def _peak_rss_kb(raw_rss: int) -> float:
    """Normalize ru_maxrss to kilobytes across Linux (KB) and macOS (bytes)."""
    if sys.platform == "darwin":
        return round(float(raw_rss) / 1024.0, 3)
    return round(float(raw_rss), 3)


def _amp_policy(use_amp: bool, device_type: str) -> tuple[bool, str | None]:
    """Return (amp_enabled, autocast_device). AMP autocast is CUDA-only in this PoC."""
    if use_amp and device_type == "cuda":
        return True, "cuda"
    return False, None


def _build_train_dataset(
    monai: dict[str, Any],
    train_rows: list[dict[str, str]],
    patch_size: list[int],
    cache_rate: float,
) -> Any:
    transform = _transforms(monai, patch_size, train=True)
    if cache_rate > 0:
        return monai["CacheDataset"](data=train_rows, transform=transform, cache_rate=float(cache_rate))
    return monai["Dataset"](data=train_rows, transform=transform)


def _run_validation_dice(
    *,
    model: Any,
    torch: Any,
    monai: dict[str, Any],
    val_rows: list[dict[str, str]],
    patch_size: list[int],
    device: Any,
    threshold: float,
) -> tuple[float, int]:
    if not val_rows:
        return 0.0, 0
    model.eval()
    roi_size = tuple(patch_size)
    dice_scores: list[float] = []
    with torch.no_grad():
        for row in val_rows:
            image = load_array(row["image"])
            label = load_array(row["label"])
            label_mask = (label == 1).astype(np.uint8)
            x = _normalize_for_inference(image)
            tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
            logits = monai["sliding_window_inference"](tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs[0].detach().cpu().numpy() >= threshold).astype(np.uint8)
            dice_scores.append(float(dice_wmh_label1(pred, label_mask)))
    return float(np.mean(dice_scores)), len(dice_scores)


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
) -> int:
    model.eval()
    roi_size = tuple(patch_size)
    count = 0
    with torch.no_grad():
        for row in val_rows:
            image = load_array(row["image"])
            x = _normalize_for_inference(image)
            tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
            logits = monai["sliding_window_inference"](tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs[0].detach().cpu().numpy() >= threshold).astype(np.uint8)
            save_array_like(row["image"], pred_dir / f"{row['case_id']}_pred.nii.gz", pred)
            count += 1
    return count


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
    image_key = str(data_cfg.get("image_key", "flair_pre_path"))
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

    train_rows = _case_rows(dataset_manifest, split_manifest, "train", image_key, label_key)
    all_val_rows = _case_rows(dataset_manifest, split_manifest, "val", image_key, label_key)
    if val_max_cases > 0:
        val_rows = all_val_rows[:val_max_cases]
        val_eval_rows = all_val_rows[:val_max_cases]
    else:
        val_rows = all_val_rows
        val_eval_rows = all_val_rows
    if not train_rows or not val_rows:
        raise ValueError("train and val rows are required for training")

    train_ds = _build_train_dataset(monai, train_rows, patch_size, cache_rate)
    train_loader = monai["DataLoader"](train_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    model = _build_model(monai, cfg)
    if device.type == "mps":
        patched_layers = apply_mps_safe_convtranspose_patch(model)
        if patched_layers == 0:
            raise RuntimeError(
                "MPS was selected but no ConvTranspose3d layers were replaced; "
                "MONAI smoke cannot run on MPS without the compatibility patch."
            )
        device_runtime = record_mps_convtranspose_patch(device_runtime, patched_layers)
    model = model.to(device)
    loss_fn = monai["DiceCELoss"](to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))

    max_epochs = int(train_cfg.get("max_epochs", 1))
    max_steps = int(train_cfg.get("max_steps_per_epoch", 2))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
    logs: list[dict[str, Any]] = []
    best_val_dice = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0
    checkpoint_path = ""
    best_checkpoint_path = str(ckpt_dir / "model_best.pt") if mode == "full" else str(ckpt_dir / "model_smoke.pt")
    amp_enabled, autocast_device = _amp_policy(use_amp, device.type)
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

        val_dice, val_n = _run_validation_dice(
            model=model,
            torch=torch,
            monai=monai,
            val_rows=val_eval_rows,
            patch_size=patch_size,
            device=device,
            threshold=threshold,
        )
        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
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
        logs.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "val_dice": val_dice,
                "val_n_cases": val_n,
                "mean_epoch_loss": mean_epoch_loss,
                "mode": mode,
            }
        )
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            if mode == "full" and bool(train_cfg.get("save_best_only", True)):
                torch.save(  # nosec B614 — local trusted best checkpoint only; not loading untrusted weights
                    {
                        "run_id": run_id,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "global_step": global_step,
                        "best_val_dice": best_val_dice,
                        "best_epoch": best_epoch,
                        "claim_boundary": "local PoC full training; not clinical or production model",
                    },
                    best_checkpoint_path,
                )
                checkpoint_path = best_checkpoint_path
        else:
            epochs_without_improvement += 1

        if mode == "full" and epochs_without_improvement >= early_stopping_patience:
            break

    if mode == "smoke" and bool(train_cfg.get("save_checkpoint", True)):
        checkpoint_path = str(ckpt_dir / "model_smoke.pt")
        torch.save(  # nosec B614 — local smoke checkpoint only; not loading untrusted weights
            {
                "run_id": run_id,
                "model_state_dict": model.state_dict(),
                "config": cfg,
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
        "best_val_dice": best_val_dice if mode == "full" else None,
        "best_epoch": best_epoch if mode == "full" else None,
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
            "policy": "cuda-only autocast; MPS/CPU run full precision in this PoC",
        },
        "resource": {"first_epoch": first_epoch_resource} if first_epoch_resource else None,
        "safety": {
            "test_split_used": False,
            "label_policy": "label==1 foreground; label==2 ignored as foreground",
            "claim_boundary": "local PoC smoke only" if mode == "smoke" else "local PoC full training only",
            "mps_execution_claim": (
                "MPS-compatible patched smoke; not native-MPS equivalence with ConvTranspose3d"
                if device.type == "mps"
                else "standard device path without MPS ConvTranspose3d patch"
            ),
        },
    }
    evidence_path = out_dir / "run_evidence.json"
    write_json(evidence_path, evidence)

    row = make_run_row(
        run_id=run_id,
        run_purpose="wmh2017_monai_smoke_training",
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
        notes=f"mode={mode}; global_step={global_step}; best_val_dice={best_val_dice:.6f}; no test110 use",
    )
    append_run_manifest(row, run_cfg.get("run_manifest", "registry/runs/run_manifest.csv"))

    print(f"Completed MONAI smoke training run_id={run_id}")
    print(f"Wrote train log: {log_path}")
    print(f"Wrote run evidence: {evidence_path}")
    if checkpoint_path:
        print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote predictions: {pred_dir}")
