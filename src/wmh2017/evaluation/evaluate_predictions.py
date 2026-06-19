"""Evaluation package for WMH2017 validation predictions.

This is local validation only. Official challenge comparison requires an
explicit official-code cross-check and reviewer approval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_wmh_label1
from wmh2017.evaluation.metric_schema import validate_case_metrics_columns
from wmh2017.evaluation.voxel_metrics import avd_wmh_label1, dice_wmh_label1, hd95_wmh_label1
from wmh2017.io.images import assert_compatible_image_geometry, load_array, load_image_metadata
from wmh2017.lineage.hashes import sha256_jsonable, sha256_path
from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown, package_versions
from wmh2017.security.path_redaction import redact_path


def _prediction_path_for_case(prediction_dir: Path, case_id: str) -> Path | None:
    candidates = [
        prediction_dir / f"{case_id}_pred.nii.gz",
        prediction_dir / f"{case_id}_pred.nii",
        prediction_dir / f"{case_id}_pred.npy",
        prediction_dir / f"{case_id}.nii.gz",
        prediction_dir / f"{case_id}.nii",
        prediction_dir / f"{case_id}.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def evaluate_predictions(
    manifest_csv: str | Path,
    split_csv: str | Path,
    prediction_dir: str | Path,
    out_dir: str | Path,
    run_id: str,
    assigned_split: str = "val",
    threshold: float = 0.5,
    strict_geometry: bool = True,
    metric_script_path: str | Path = "src/wmh2017/evaluation/evaluate_predictions.py",
    model_artifact_path: str | Path = "",
    config_path: str | Path = "",
    skip_missing_predictions: bool = False,
    prediction_manifest_path: str | Path = "",
) -> dict[str, Any]:
    manifest_csv = Path(manifest_csv)
    split_csv = Path(split_csv)
    prediction_dir = Path(prediction_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    cases = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    if cases.empty:
        raise ValueError(f"no cases assigned to split={assigned_split} in {split_csv}")

    created_at = datetime.now(timezone.utc).isoformat()
    model_hash = sha256_path(model_artifact_path) if model_artifact_path else ""
    config_hash = sha256_path(config_path) if config_path else ""
    pred_manifest_hash = sha256_path(prediction_manifest_path) if prediction_manifest_path else ""

    records: list[dict[str, Any]] = []
    for _, srow in cases.iterrows():
        case_id = str(srow["case_id"])
        mrow = manifest[manifest["case_id"].astype(str) == case_id]
        if mrow.empty:
            raise ValueError(f"case_id={case_id} exists in split but not manifest")
        m = mrow.iloc[0]
        challenge_split = str(m.get("challenge_split", "")).lower()
        if challenge_split == "test":
            raise ValueError(
                f"case_id={case_id} belongs to challenge_split=test; "
                "test split must not be used for local validation, threshold tuning, "
                "model selection, or early stopping"
            )
        label_path = str(m.get("wmh_path", "") or m.get("mask_path", "") or "")
        if not label_path:
            raise ValueError(f"case_id={case_id} has no label path; local validation requires labels")

        pred_path = _prediction_path_for_case(prediction_dir, case_id)
        if pred_path is None:
            if skip_missing_predictions:
                continue
            raise FileNotFoundError(f"prediction not found for case_id={case_id} in {prediction_dir}")
        pred_meta = load_image_metadata(pred_path)
        label_meta = load_image_metadata(label_path)
        assert_compatible_image_geometry(
            pred_meta,
            label_meta,
            case_id=case_id,
            require_affine_match=strict_geometry,
            require_spacing_match=strict_geometry,
        )

        pred = load_array(pred_path)
        label = load_array(label_path)
        pred_mask = (pred >= threshold).astype(np.uint8)
        label_mask = np.asarray(label).astype(np.uint8)
        spacing = label_meta.spacing or None
        lesion = lesion_recall_f1_wmh_label1(pred_mask, label_mask)

        site_or_center = str(srow.get("site", "") or m.get("site", "") or "")
        record = {
            "run_id": run_id,
            "case_id": case_id,
            "assigned_split": assigned_split,
            "site_or_center": site_or_center,
            "prediction_path_redacted": redact_path(pred_path),
            "prediction_sha256": sha256_path(pred_path),
            "label_path_redacted": redact_path(label_path),
            "label_sha256": sha256_path(label_path),
            **pred_meta.to_record("prediction"),
            **label_meta.to_record("label"),
            "geometry_policy": "shape+spacing+affine" if strict_geometry else "shape_only",
            "threshold": threshold,
            "dice": float(dice_wmh_label1(pred_mask, label_mask)),
            "hd95": float(hd95_wmh_label1(pred_mask, label_mask, spacing=spacing)),
            "avd": float(avd_wmh_label1(pred_mask, label_mask, spacing=spacing)),
            "lesion_recall": float(lesion["lesion_recall"]),
            "lesion_f1": float(lesion["lesion_f1"]),
            "target_lesion_count": int(lesion.get("target_lesion_count", lesion.get("n_target", 0))),
            "pred_lesion_count": int(lesion.get("pred_lesion_count", lesion.get("n_pred", 0))),
            "split_manifest_sha256": sha256_path(split_csv),
            "model_artifact_sha256": model_hash,
            "metric_script_sha256": sha256_path(metric_script_path),
            "config_sha256": config_hash,
            "code_commit": git_commit_or_unknown(),
            "created_at_utc": created_at,
        }
        records.append(record)

    if not records:
        raise ValueError(
            f"no cases evaluated for split={assigned_split}; " f"predictions missing under {prediction_dir}"
        )

    result_df = pd.DataFrame(records)
    missing = validate_case_metrics_columns(result_df.columns.tolist())
    if missing:
        raise ValueError(f"case_metrics missing required columns: {missing}")

    case_csv = out_dir / "case_metrics.csv"
    result_df.to_csv(case_csv, index=False)

    summary = {
        "run_id": run_id,
        "assigned_split": assigned_split,
        "n_cases": int(len(result_df)),
        "threshold": threshold,
        "strict_geometry": bool(strict_geometry),
        "geometry_policy": "shape+spacing+affine" if strict_geometry else "shape_only",
        "geometry_metrics_physical_units": bool(strict_geometry),
        "mean_dice": float(result_df["dice"].mean()),
        "median_dice": float(result_df["dice"].median()),
        "mean_hd95": float(result_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_avd": float(result_df["avd"].mean()),
        "mean_lesion_recall": float(result_df["lesion_recall"].mean()),
        "mean_lesion_f1": float(result_df["lesion_f1"].mean()),
        "case_metrics_csv": str(case_csv),
        "case_metrics_sha256": sha256_path(case_csv),
        "dataset_manifest": str(manifest_csv),
        "dataset_manifest_hash": sha256_path(manifest_csv),
        "split_manifest": str(split_csv),
        "split_manifest_hash": sha256_path(split_csv),
        "prediction_dir": str(prediction_dir),
        "model_artifact_hash": model_hash,
        "config_hash": config_hash,
        "prediction_manifest_hash": pred_manifest_hash,
        "metric_script_hash": sha256_path(metric_script_path),
        "code_commit": git_commit_or_unknown(),
        "package_versions": package_versions(),
        "official_claim_status": {
            "local_metrics_available": True,
            "official_evaluator_parity_complete": False,
            "leaderboard_claim_allowed": False,
        },
        "claim_allowed": {
            "local_validation": True,
            "official_comparable": False,
            "leaderboard_or_sota": False,
        },
        "claim_boundary": "local validation only; no diagnostic, customer, leaderboard, or SOTA claim without official evaluation cross-check and review",
    }
    summary["summary_hash"] = sha256_jsonable(summary)

    summary_json = out_dir / "metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
