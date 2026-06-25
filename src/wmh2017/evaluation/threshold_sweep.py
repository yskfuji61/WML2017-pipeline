"""Validation-only threshold sweep over saved probability maps."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.data.split_guard import guard_challenge_split_test
from wmh2017.evaluation.lesion_metrics import (
    DEFAULT_SIZE_BINS,
    lesion_recall_by_size_bins_wmh_label1,
    lesion_recall_f1_wmh_label1,
)
from wmh2017.evaluation.postprocess import post_process_binary
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
        guard_challenge_split_test(
            case_id, str(m.get("challenge_split", "")), assigned_split=assigned_split, context="threshold_sweep"
        )
        label_path = str(m.get("wmh_path", "") or m.get("mask_path", "") or "")
        if not label_path:
            raise ValueError(f"case_id={case_id} has no label path")
        image_path = str(m.get("flair_pre_path", "") or m.get("flair_path", "") or m.get("image_path", "") or "")
        rows.append({"case_id": case_id, "label_path": label_path, "image_path": image_path})
    return rows


def _threshold_prediction(probs: np.ndarray, threshold: float, min_component_size: int) -> np.ndarray:
    """Binarize at ``threshold``; apply CC min-size filter only when requested.

    With ``min_component_size == 0`` this is byte-identical to ``probs >= threshold``.
    """
    if min_component_size > 0:
        return post_process_binary(probs, threshold=float(threshold), min_size=int(min_component_size))
    return (probs >= float(threshold)).astype(np.uint8)


def _metrics_at_threshold(
    *,
    probs: np.ndarray,
    label_mask: np.ndarray,
    threshold: float,
    spacing: tuple[float, ...] | None,
    min_component_size: int = 0,
) -> dict[str, float]:
    pred_mask = _threshold_prediction(probs, threshold, min_component_size)
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
    min_component_size: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep thresholds on validation probability maps.

    ``min_component_size`` (default 0 = off) optionally applies a connected-component
    min-size filter at each threshold; with 0 the output is byte-identical to before.

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
                min_component_size=min_component_size,
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


def select_best_threshold(
    summary_df: pd.DataFrame,
    *,
    selection_metric: str = "mean_dice",
    tie_breakers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select best threshold by ``selection_metric`` with tie-breakers.

    Default behavior is backward compatible: max ``mean_dice`` on val, tie-broken by
    ``mean_lesion_recall`` then ``mean_lesion_f1``. This selects a threshold for the
    already-fixed checkpoint; it never changes which checkpoint was selected.
    """
    if summary_df.empty:
        raise ValueError("threshold sweep summary is empty")
    if tie_breakers is None:
        tie_breakers = ("mean_lesion_recall", "mean_lesion_f1")
    sort_cols = [selection_metric, *tie_breakers]
    missing = [c for c in sort_cols if c not in summary_df.columns]
    if missing:
        raise KeyError(f"threshold sweep summary missing columns: {missing}")
    ranked = summary_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
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
        "selection_policy": (f"max {selection_metric} on val; tie-break " + " then ".join(tie_breakers)),
        "threshold_selection_metric": selection_metric,
        "threshold_tie_breakers": list(tie_breakers),
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
    checkpoint_selection_metric: str = "mean_dice",
    min_component_size: int = 0,
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
        # Threshold-best is independent from checkpoint-best: choosing this threshold
        # does not change which checkpoint was selected during training.
        "threshold_best_is_checkpoint_best": False,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "threshold_selection_metric": best.get("threshold_selection_metric", "mean_dice"),
        "threshold_tie_breakers": best.get("threshold_tie_breakers", ["mean_lesion_recall", "mean_lesion_f1"]),
        "allowed_use": "validation-only threshold analysis",
        "prohibited_use": [
            "test threshold tuning",
            "SOTA claim",
            "clinical decision",
            "production deployment",
        ],
        "best": best,
        "summary_csv": str(summary_csv),
        "per_case_csv": str(per_case_csv),
    }
    if min_component_size > 0:
        payload["post_process"] = {"min_component_size": int(min_component_size)}
    payload["artifact_hash"] = sha256_jsonable(payload)
    best_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _aggregate_bin_recall(
    accumulator: dict[str, dict[str, int]],
    bin_rows: list[dict[str, float | int | str | None]],
) -> None:
    for row in bin_rows:
        name = str(row["bin"])
        acc = accumulator.setdefault(name, {"n_target": 0, "n_detected": 0})
        acc["n_target"] += cast(int, row["n_target"])
        acc["n_detected"] += cast(int, row["n_detected"])


def lesion_size_bin_audit(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    probs_dir: str | Path,
    threshold: float,
    min_component_size: int,
    assigned_split: str = "val",
    bins: Sequence[tuple[str, int, int | None]] = DEFAULT_SIZE_BINS,
    connectivity: int = 26,
) -> dict[str, Any]:
    """Compare baseline vs post-processed lesion recall per GT size bin (val only).

    For each case, recall-by-size-bin is micro-aggregated (sum detected / sum target across
    cases) for the baseline prediction (``probs >= threshold``) and the post-processed
    prediction (CC min-size filter). A drop in the smallest non-empty bin sets
    ``small_lesion_recall_regressed`` — the signal that component removal helped Dice by
    deleting true small lesions. Diagnostic only; never tunes on the test split.
    """
    probs_dir = Path(probs_dir)
    case_rows = _case_rows_for_split(manifest_csv, split_csv, assigned_split)

    baseline_acc: dict[str, dict[str, int]] = {}
    post_acc: dict[str, dict[str, int]] = {}
    bin_order = [name for name, _, _ in bins]
    bin_meta = {name: (lo, hi) for name, lo, hi in bins}

    for row in case_rows:
        probs = _load_probability_map(probs_dir, row["case_id"])
        label_mask = wmh_foreground_mask(load_array(row["label_path"])).astype(np.uint8)
        baseline_pred = _threshold_prediction(probs, threshold, 0)
        post_pred = _threshold_prediction(probs, threshold, min_component_size)
        _aggregate_bin_recall(baseline_acc, lesion_recall_by_size_bins_wmh_label1(baseline_pred, label_mask, bins=bins))
        _aggregate_bin_recall(post_acc, lesion_recall_by_size_bins_wmh_label1(post_pred, label_mask, bins=bins))

    def _recall(acc: dict[str, int]) -> float | None:
        return (acc["n_detected"] / acc["n_target"]) if acc["n_target"] else None

    eps = 1e-9
    per_bin: list[dict[str, Any]] = []
    regressed = False
    smallest_seen = False
    for name in bin_order:
        base = baseline_acc.get(name, {"n_target": 0, "n_detected": 0})
        post = post_acc.get(name, {"n_target": 0, "n_detected": 0})
        base_recall = _recall(base)
        post_recall = _recall(post)
        delta = None if (base_recall is None or post_recall is None) else (post_recall - base_recall)
        lo, hi = bin_meta[name]
        per_bin.append(
            {
                "bin": name,
                "min_voxels": int(lo),
                "max_voxels": (None if hi is None else int(hi)),
                "n_target": int(base["n_target"]),
                "baseline_recall": base_recall,
                "post_recall": post_recall,
                "delta_recall": delta,
            }
        )
        # The smallest non-empty bin is the small-lesion regression sentinel.
        if not smallest_seen and base["n_target"] > 0:
            smallest_seen = True
            if delta is not None and delta < -eps:
                regressed = True

    return {
        "threshold": float(threshold),
        "min_component_size": int(min_component_size),
        "connectivity": int(connectivity),
        "assigned_split": assigned_split,
        "n_cases": len(case_rows),
        "per_bin": per_bin,
        "small_lesion_recall_regressed": regressed,
    }


def write_lesion_size_bin_audit_artifact(
    *,
    out_dir: str | Path,
    audit: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Write lesion_size_bin_audit.json with claim guards + artifact hash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        **audit,
        # Post-processing is a validation diagnostic; it does not change the selected
        # checkpoint and must never be tuned on the test split.
        "threshold_best_is_checkpoint_best": False,
        "allowed_use": "validation-only postprocess audit",
        "prohibited_use": [
            "test threshold tuning",
            "SOTA claim",
            "clinical decision",
            "production deployment",
        ],
        "claim_boundary": "local validation lesion-size diagnostic; not SOTA/official/clinical/production",
    }
    payload["artifact_hash"] = sha256_jsonable(payload)
    (out_dir / "lesion_size_bin_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
