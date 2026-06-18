#!/usr/bin/env python3
"""Validate case_metrics.csv against v2 metric table contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wmh2017.evaluation.metric_schema import validate_case_metrics_columns
from wmh2017.lineage.hashes import sha256_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate metric table contract.")
    parser.add_argument("case_metrics_csv")
    parser.add_argument("--prediction-dir", default="")
    args = parser.parse_args()

    path = Path(args.case_metrics_csv)
    if not path.exists():
        raise SystemExit(f"missing case metrics: {path}")

    df = pd.read_csv(path)
    failures: list[str] = []

    missing = validate_case_metrics_columns(df.columns.tolist())
    if missing:
        failures.append(f"missing columns: {missing}")

    for col in ("run_id", "prediction_sha256", "label_sha256", "code_commit"):
        if col in df.columns and df[col].astype(str).str.strip().eq("").any():
            failures.append(f"empty values in required column: {col}")

    if "threshold" in df.columns and df["threshold"].isna().any():
        failures.append("threshold contains empty values")

    if "case_id" in df.columns and "assigned_split" in df.columns:
        dup = df.duplicated(subset=["case_id", "assigned_split"], keep=False)
        if dup.any():
            failures.append(f"duplicate case_id within split: {df.loc[dup, 'case_id'].tolist()}")

    for metric in ("dice", "lesion_recall", "lesion_f1"):
        if metric in df.columns:
            if (df[metric] < -1e-6).any() or (df[metric] > 1.0 + 1e-6).any():
                failures.append(f"{metric} outside [0,1] range")

    for metric in ("hd95", "avd"):
        if metric in df.columns and (df[metric] < -1e-6).any():
            failures.append(f"{metric} must be non-negative")

    if args.prediction_dir and "case_id" in df.columns and "prediction_sha256" in df.columns:
        pred_dir = Path(args.prediction_dir)
        for _, row in df.iterrows():
            case_id = str(row["case_id"])
            pred_candidates = [
                pred_dir / f"{case_id}_pred.nii.gz",
                pred_dir / f"{case_id}_pred.npy",
            ]
            pred_path = next((p for p in pred_candidates if p.exists()), None)
            if pred_path is None:
                continue
            actual = sha256_path(pred_path)
            if actual and str(row["prediction_sha256"]) != actual:
                failures.append(f"prediction hash mismatch for {case_id}")

    if failures:
        raise SystemExit("metric table validation FAIL:\n" + "\n".join(failures))
    print(f"metric table validation PASS: {path}")


if __name__ == "__main__":
    main()
