"""Prediction manifest and prediction-label linkage for run lineage."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from wmh2017.lineage.hashes import sha256_path, write_hash_sidecar, write_named_sidecar
from wmh2017.security.path_redaction import redact_path


def build_prediction_manifest(
    *,
    run_id: str,
    prediction_dir: Path,
    manifest_csv: Path,
    split_csv: Path,
    assigned_split: str = "val",
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    cases = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()]
    rows: list[dict[str, Any]] = []
    for _, srow in cases.iterrows():
        case_id = str(srow["case_id"])
        mrow = manifest[manifest["case_id"].astype(str) == case_id]
        if mrow.empty:
            continue
        m = mrow.iloc[0]
        label_path = str(m.get("wmh_path", "") or m.get("mask_path", "") or "")
        pred_candidates = [
            prediction_dir / f"{case_id}_pred.nii.gz",
            prediction_dir / f"{case_id}_pred.nii",
        ]
        pred_path = next((p for p in pred_candidates if p.exists()), None)
        if pred_path is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "assigned_split": assigned_split,
                "prediction_path_redacted": redact_path(pred_path),
                "prediction_sha256": sha256_path(pred_path),
                "label_path_redacted": redact_path(label_path),
                "label_sha256": sha256_path(label_path) if label_path else "",
            }
        )
        write_hash_sidecar(pred_path)
    return rows


def write_prediction_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("run_id,case_id,assigned_split,prediction_path_redacted,prediction_sha256,label_path_redacted,label_sha256\n", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_named_sidecar(path, "prediction_manifest.sha256")
    return path


def write_prediction_label_linkage(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    linkage_rows = [
        {
            "run_id": r["run_id"],
            "case_id": r["case_id"],
            "prediction_sha256": r["prediction_sha256"],
            "label_sha256": r["label_sha256"],
        }
        for r in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "case_id", "prediction_sha256", "label_sha256"],
        )
        writer.writeheader()
        writer.writerows(linkage_rows)
    write_hash_sidecar(path)
    return path
