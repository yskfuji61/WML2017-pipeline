"""Export foreground probability maps for validation threshold sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wmh2017.data.preprocessing import normalize_nonzero_channelwise
from wmh2017.io.images import load_array, save_array_like
from wmh2017.training.mps_compat import (
    apply_mps_safe_convtranspose_patch,
    enable_mps_cpu_fallback,
    resolve_training_device,
)


def infer_foreground_probability(
    *,
    model: Any,
    torch: Any,
    monai: dict[str, Any],
    image_path: str,
    patch_size: list[int] | tuple[int, ...],
    device: Any,
) -> np.ndarray:
    """Run sliding-window inference and return foreground probability (Z, Y, X) float32."""
    image = load_array(image_path)
    x = normalize_nonzero_channelwise(image)
    tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
    roi_size = tuple(patch_size)
    model.eval()
    with torch.no_grad():
        logits = monai["sliding_window_inference"](tensor, roi_size=roi_size, sw_batch_size=1, predictor=model)
        probs = torch.softmax(logits, dim=1)[:, 1]
    return probs[0].detach().cpu().numpy().astype(np.float32)


def save_case_probability_map(probs: np.ndarray, out_path: str | Path) -> None:
    """Persist probability map as compressed npz (key=probs, float32)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), probs=np.asarray(probs, dtype=np.float32))


def save_case_prediction(
    *,
    probs: np.ndarray,
    threshold: float,
    reference_image_path: str,
    pred_path: str | Path,
    prob_path: str | Path | None = None,
) -> None:
    """Save binary prediction and optional probability map."""
    pred = (probs >= float(threshold)).astype(np.uint8)
    save_array_like(reference_image_path, pred_path, pred)
    if prob_path is not None:
        save_case_probability_map(probs, prob_path)


def load_model_from_checkpoint(
    *,
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    device: Any,
    torch: Any,
    monai: dict[str, Any],
) -> Any:
    """Build MONAI UNet, apply MPS patch if needed, and load checkpoint weights."""
    from wmh2017.training.train_monai import _build_model

    model = _build_model(monai, cfg)
    if device.type == "mps":
        apply_mps_safe_convtranspose_patch(model)
    state = torch.load(  # nosec B614 — local checkpoint only
        str(checkpoint_path), map_location=device, weights_only=False
    )
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)


def export_val_probabilities(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    probs_dir: str | Path | None = None,
    assigned_split: str = "val",
    save_binary_predictions: bool = False,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Export val probability maps from a trained checkpoint (no retraining)."""
    from wmh2017.training.train_monai import (
        _case_rows,
        _load_config,
        _require_monai_stack,
        _set_seed,
    )

    enable_mps_cpu_fallback()
    torch, monai = _require_monai_stack()
    cfg = _load_config(config_path)

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    effective_threshold = float(threshold if threshold is not None else cfg.get("training", {}).get("threshold", 0.5))

    run_id = str(run_cfg.get("run_id", "wmh2017_export"))
    seed = int(run_cfg.get("seed", 42))
    _set_seed(seed, torch)
    device, _device_runtime = resolve_training_device(torch, str(run_cfg.get("device", "auto")))

    dataset_manifest = str(data_cfg["dataset_manifest"])
    split_manifest = str(data_cfg["split_manifest"])
    image_key = str(data_cfg.get("image_key", "flair_pre_path"))
    label_key = str(data_cfg.get("label_key", "wmh_path"))
    patch_size = list(data_cfg.get("patch_size", [32, 32, 32]))
    val_max_cases = int(data_cfg.get("val_max_cases", 0))

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    pred_dir = out_dir / "predictions"
    target_probs_dir = Path(probs_dir) if probs_dir is not None else pred_dir / "probs"
    target_probs_dir.mkdir(parents=True, exist_ok=True)
    if save_binary_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    val_rows = _case_rows(dataset_manifest, split_manifest, assigned_split, image_key, label_key)
    if val_max_cases > 0:
        val_rows = val_rows[:val_max_cases]
    if not val_rows:
        raise ValueError(f"no val rows for assigned_split={assigned_split}")

    model = load_model_from_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        torch=torch,
        monai=monai,
    )

    exported: list[str] = []
    for row in val_rows:
        probs = infer_foreground_probability(
            model=model,
            torch=torch,
            monai=monai,
            image_path=row["image"],
            patch_size=patch_size,
            device=device,
        )
        prob_path = target_probs_dir / f"{row['case_id']}.npz"
        save_case_probability_map(probs, prob_path)
        exported.append(str(prob_path))
        if save_binary_predictions:
            save_case_prediction(
                probs=probs,
                threshold=effective_threshold,
                reference_image_path=row["image"],
                pred_path=pred_dir / f"{row['case_id']}_pred.nii.gz",
            )

    return {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "probs_dir": str(target_probs_dir),
        "n_cases": len(exported),
        "prob_paths": exported,
        "assigned_split": assigned_split,
    }
