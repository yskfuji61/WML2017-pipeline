"""Cross-architecture probability fusion for WMH2017 (active port; val-only tuning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_wmh_label1
from wmh2017.evaluation.threshold_sweep import default_threshold_grid
from wmh2017.evaluation.voxel_metrics import avd_wmh_label1, dice_wmh_label1, hd95_wmh_label1
from wmh2017.io.images import load_array, load_image_metadata


def load_probability_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(str(path))
        if "probs" not in data:
            raise ValueError(f"npz missing probs key: {path}")
        return np.asarray(data["probs"], dtype=np.float32)
    arr = load_array(path)
    return np.asarray(arr, dtype=np.float32)


def fuse_probability_maps(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    secondary_weight: float = 0.5,
) -> np.ndarray:
    """Fuse two same-shape probability maps with convex weighting."""
    w = float(secondary_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"secondary_weight must be in [0,1], got {w}")
    if primary.shape != secondary.shape:
        raise ValueError(f"shape mismatch: {primary.shape} vs {secondary.shape}")
    return ((1.0 - w) * primary + w * secondary).astype(np.float32)


def post_process_binary(
    prob: np.ndarray,
    *,
    threshold: float,
    min_size: int = 0,
    adaptive_low_thr: float = 0.0,
    adaptive_high_vol: int = 0,
) -> np.ndarray:
    """Threshold + optional CC filter + optional adaptive low-threshold rescue."""
    binary = (prob >= float(threshold)).astype(np.uint8)
    if adaptive_low_thr > 0 and adaptive_high_vol > 0 and int(binary.sum()) > adaptive_high_vol:
        binary = (prob >= float(adaptive_low_thr)).astype(np.uint8)
    if min_size > 0 and binary.sum() > 0:
        lbl, n = cc_label(binary)
        keep = np.zeros_like(binary)
        for i in range(1, n + 1):
            comp = lbl == i
            if comp.sum() >= min_size:
                keep[comp] = 1
        binary = keep
    return binary.astype(np.uint8)


def _case_rows(manifest_csv: str | Path, split_csv: str | Path, assigned_split: str) -> list[dict[str, str]]:
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    rows: list[dict[str, str]] = []
    for _, srow in split.iterrows():
        case_id = str(srow["case_id"])
        mrow = manifest[manifest["case_id"].astype(str) == case_id]
        if mrow.empty:
            raise ValueError(f"case_id={case_id} missing in manifest")
        m = mrow.iloc[0]
        if str(m.get("challenge_split", "")).lower() == "test":
            raise ValueError(f"test case {case_id} cannot be used for ensemble threshold tuning")
        label_path = str(m.get("wmh_path", "") or m.get("mask_path", "") or "")
        rows.append({"case_id": case_id, "label_path": label_path})
    return rows


def evaluate_fused_predictions(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    primary_probs_dir: str | Path,
    secondary_probs_dir: str | Path,
    assigned_split: str = "val",
    secondary_weight: float = 0.5,
    threshold: float = 0.5,
    min_size: int = 0,
    adaptive_low_thr: float = 0.0,
    adaptive_high_vol: int = 0,
) -> dict[str, Any]:
    """Fuse two prob dirs and evaluate MICCAI metrics at a fixed threshold."""
    primary_probs_dir = Path(primary_probs_dir)
    secondary_probs_dir = Path(secondary_probs_dir)
    records: list[dict[str, float | str]] = []
    for row in _case_rows(manifest_csv, split_csv, assigned_split):
        case_id = row["case_id"]
        p_path = primary_probs_dir / f"{case_id}.npz"
        s_path = secondary_probs_dir / f"{case_id}.npz"
        if not p_path.exists() or not s_path.exists():
            raise FileNotFoundError(f"missing fused inputs for case_id={case_id}")
        fused = fuse_probability_maps(
            load_probability_volume(p_path),
            load_probability_volume(s_path),
            secondary_weight=secondary_weight,
        )
        label = load_array(row["label_path"])
        label_mask = wmh_foreground_mask(label).astype(np.uint8)
        spacing_raw = load_image_metadata(row["label_path"]).spacing
        spacing = spacing_raw if spacing_raw and len(spacing_raw) >= 3 else None
        pred = post_process_binary(
            fused,
            threshold=threshold,
            min_size=min_size,
            adaptive_low_thr=adaptive_low_thr,
            adaptive_high_vol=adaptive_high_vol,
        )
        lesion = lesion_recall_f1_wmh_label1(pred, label_mask)
        records.append(
            {
                "case_id": case_id,
                "dice": float(dice_wmh_label1(pred, label_mask)),
                "hd95": float(hd95_wmh_label1(pred, label_mask, spacing=spacing)),
                "avd": float(avd_wmh_label1(pred, label_mask, spacing=spacing)),
                "lesion_recall": float(lesion["lesion_recall"]),
                "lesion_f1": float(lesion["lesion_f1"]),
            }
        )
    df = pd.DataFrame(records)
    return {
        "n_cases": int(len(df)),
        "secondary_weight": float(secondary_weight),
        "threshold": float(threshold),
        "mean_dice": float(df["dice"].mean()),
        "mean_lesion_recall": float(df["lesion_recall"].mean()),
        "mean_lesion_f1": float(df["lesion_f1"].mean()),
        "mean_hd95": float(df["hd95"].mean()),
        "mean_avd": float(df["avd"].mean()),
        "per_case": df,
    }


def sweep_ensemble_hyperparams(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    primary_probs_dir: str | Path,
    secondary_probs_dir: str | Path,
    assigned_split: str = "val",
    secondary_weights: list[float] | None = None,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Grid search secondary weight and threshold on val (max mean_dice)."""
    weights = secondary_weights or [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = thresholds or default_threshold_grid()
    rows: list[dict[str, Any]] = []
    for w in weights:
        for thr in grid:
            summary = evaluate_fused_predictions(
                manifest_csv=manifest_csv,
                split_csv=split_csv,
                primary_probs_dir=primary_probs_dir,
                secondary_probs_dir=secondary_probs_dir,
                assigned_split=assigned_split,
                secondary_weight=float(w),
                threshold=float(thr),
            )
            rows.append(
                {
                    "secondary_weight": float(w),
                    "threshold": float(thr),
                    "mean_dice": summary["mean_dice"],
                    "mean_lesion_recall": summary["mean_lesion_recall"],
                    "mean_lesion_f1": summary["mean_lesion_f1"],
                }
            )
    return pd.DataFrame(rows).sort_values(["mean_dice", "mean_lesion_recall"], ascending=[False, False])
