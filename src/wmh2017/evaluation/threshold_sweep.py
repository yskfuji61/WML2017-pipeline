"""Validation-only threshold sweep over saved probability maps."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_wmh_label1
from wmh2017.evaluation.voxel_metrics import avd_wmh_label1, dice_wmh_label1, hd95_wmh_label1
from wmh2017.io.images import load_array, load_image_metadata
from wmh2017.lineage.hashes import sha256_jsonable


def default_threshold_grid(*, start: float = 0.05, stop: float = 0.50, step: float = 0.05) -> list[float]:
    """Return default validation threshold grid."""
    values = np.arange(start, stop + step * 0.5, step)
    return [round(float(v), 4) for v in values.tolist()]


def _load_probability_map(probs_dir: Path, case_id: str) -> np.ndarray:
    path = probs_dir / f"{case_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"probability map not found for case_id={case_id}: {path}")
    data = np.load(str(path))
    if "probs" not in data:
        raise ValueError(f"probability npz missing 'probs' key: {path}")
    return np.asarray(data["probs"], dtype=np.float32)


def _case_rows_for_split(
    manifest_csv: str | Path,
    split_csv: str | Path,
    assigned_split: str,
) -> list[dict[str, str]]:
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    if split.empty:
        raise ValueError(f"no rows for assigned_split={assigned_split} in {split_csv}")

    rows: list[dict[str, str]] = []
    for _, srow in split.iterrows():
        case_id = str(srow["case_id"])
        mrow = manifest[manifest["case_id"].astype(str) == case_id]
        if mrow.empty:
            raise ValueError(f"case_id={case_id} exists in split but not manifest")
        m = mrow.iloc[0]
        if str(m.get("challenge_split", "")).lower() == "test":
            raise ValueError(
                f"case_id={case_id} belongs to challenge_split=test; "
                "test split must not be used for threshold tuning"
            )
        label_path = str(m.get("wmh_path", "") or m.get("mask_path", "") or "")
        if not label_path:
            raise ValueError(f"case_id={case_id} has no label path")
        image_path = str(m.get("flair_pre_path", "") or m.get("flair_path", "") or m.get("image_path", "") or "")
        rows.append({"case_id": case_id, "label_path": label_path, "image_path": image_path})
    return rows


def _metrics_at_threshold(
    *,
    probs: np.ndarray,
    label_mask: np.ndarray,
    threshold: float,
    spacing: tuple[float, ...] | None,
) -> dict[str, float]:
    pred_mask = (probs >= float(threshold)).astype(np.uint8)
    lesion = lesion_recall_f1_wmh_label1(pred_mask, label_mask)
    hd95 = hd95_wmh_label1(pred_mask, label_mask, spacing=spacing)
    return {
        "threshold": float(threshold),
        "dice": float(dice_wmh_label1(pred_mask, label_mask)),
        "hd95": float(hd95),
        "avd": float(avd_wmh_label1(pred_mask, label_mask, spacing=spacing)),
        "lesion_recall": float(lesion["lesion_recall"]),
        "lesion_f1": float(lesion["lesion_f1"]),
    }


def sweep_thresholds(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    probs_dir: str | Path,
    assigned_split: str = "val",
    thresholds: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep thresholds on validation probability maps.

    Returns:
        summary_df: one row per threshold with mean metrics across cases
        per_case_df: one row per (case_id, threshold)
    """
    probs_dir = Path(probs_dir)
    case_rows = _case_rows_for_split(manifest_csv, split_csv, assigned_split)
    grid = list(thresholds) if thresholds is not None else default_threshold_grid()

    per_case_records: list[dict[str, Any]] = []
    for row in case_rows:
        case_id = row["case_id"]
        probs = _load_probability_map(probs_dir, case_id)
        label = load_array(row["label_path"])
        label_mask = wmh_foreground_mask(label).astype(np.uint8)
        label_meta = load_image_metadata(row["label_path"])
        spacing = label_meta.spacing or None
        for thr in grid:
            metrics = _metrics_at_threshold(
                probs=probs,
                label_mask=label_mask,
                threshold=thr,
                spacing=spacing,
            )
            per_case_records.append({"case_id": case_id, **metrics})

    per_case_df = pd.DataFrame(per_case_records)
    summary_df = (
        per_case_df.groupby("threshold", as_index=False)
        .agg(
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            mean_hd95=("hd95", "mean"),
            mean_avd=("avd", "mean"),
            mean_lesion_recall=("lesion_recall", "mean"),
            mean_lesion_f1=("lesion_f1", "mean"),
            n_cases=("case_id", "nunique"),
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )
    return summary_df, per_case_df


def select_best_threshold(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Select best threshold by mean_dice, tie-break with mean_lesion_recall."""
    if summary_df.empty:
        raise ValueError("threshold sweep summary is empty")
    ranked = summary_df.sort_values(
        ["mean_dice", "mean_lesion_recall", "mean_lesion_f1"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "mean_dice": float(best["mean_dice"]),
        "median_dice": float(best["median_dice"]),
        "mean_hd95": float(best["mean_hd95"]),
        "mean_avd": float(best["mean_avd"]),
        "mean_lesion_recall": float(best["mean_lesion_recall"]),
        "mean_lesion_f1": float(best["mean_lesion_f1"]),
        "n_cases": int(best["n_cases"]),
        "selection_policy": "max mean_dice on val; tie-break lesion_recall then lesion_f1",
        "sweep_split": "val",
    }


def write_threshold_sweep_artifacts(
    *,
    out_dir: str | Path,
    summary_df: pd.DataFrame,
    per_case_df: pd.DataFrame,
    best: dict[str, Any],
    run_id: str,
    probs_dir: str | Path,
    training_threshold: float,
) -> dict[str, Any]:
    """Write sweep CSV/JSON artifacts under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "threshold_sweep_results.csv"
    per_case_csv = out_dir / "threshold_sweep_per_case.csv"
    best_json = out_dir / "threshold_sweep_best.json"

    summary_df.to_csv(summary_csv, index=False)
    per_case_df.to_csv(per_case_csv, index=False)

    payload = {
        "run_id": run_id,
        "probs_dir": str(probs_dir),
        "training_threshold": float(training_threshold),
        "threshold_policy": {
            "training_threshold": float(training_threshold),
            "sweep_best_threshold": float(best["threshold"]),
            "sweep_split": "val",
            "selection_policy": best["selection_policy"],
        },
        "best": best,
        "summary_csv": str(summary_csv),
        "per_case_csv": str(per_case_csv),
    }
    payload["artifact_hash"] = sha256_jsonable(payload)
    best_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_binary_predictions_at_threshold(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    probs_dir: str | Path,
    prediction_dir: str | Path,
    threshold: float,
    assigned_split: str = "val",
) -> int:
    """Write binary predictions from probability maps at a fixed threshold."""
    from wmh2017.inference.export_probabilities import save_case_prediction

    probs_dir = Path(probs_dir)
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _case_rows_for_split(manifest_csv, split_csv, assigned_split)
    count = 0
    for row in case_rows:
        probs = _load_probability_map(probs_dir, row["case_id"])
        image_path = row.get("image_path") or row["label_path"]
        save_case_prediction(
            probs=probs,
            threshold=threshold,
            reference_image_path=image_path,
            pred_path=prediction_dir / f"{row['case_id']}_pred.nii.gz",
        )
        count += 1
    return count
