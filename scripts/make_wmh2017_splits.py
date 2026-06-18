#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.data.splits import SplitPolicy, assert_no_test_contamination, make_train_val_split


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/validation split from challenge training cases only with test-contamination guard."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--source-split-col", default="challenge_split")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_id = f"WMH2017-TRAIN-VAL-SEED{args.seed}"
    split_df = make_train_val_split(
        manifest,
        SplitPolicy(
            split_id=split_id,
            seed=args.seed,
            train_ratio=args.train_ratio,
            source_split_col=args.source_split_col,
        ),
    )
    assert_no_test_contamination(split_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_csv = out_dir / f"wmh2017_train_val_seed{args.seed}.csv"
    heldout_csv = out_dir / "wmh2017_test110_heldout.csv"
    summary_json = out_dir / f"split_summary_seed{args.seed}.json"

    train_val = split_df[split_df["assigned_split"].isin(["train", "val"])].copy()
    heldout = split_df[split_df["assigned_split"] == "heldout_eval"].copy()
    train_val.to_csv(split_csv, index=False)
    heldout.to_csv(heldout_csv, index=False)

    site_counts = (
        split_df.groupby(["assigned_split", "site"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    scanner_counts = (
        split_df.groupby(["assigned_split", "scanner_code"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    summary = {
        "split_id": split_id,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "counts": split_df["assigned_split"].value_counts().to_dict(),
        "site_counts": site_counts,
        "scanner_counts": scanner_counts,
        "train_val_csv": str(split_csv),
        "train_val_csv_sha256": sha256_file(split_csv),
        "heldout_csv": str(heldout_csv),
        "heldout_csv_sha256": sha256_file(heldout_csv),
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256_file(Path(args.manifest)),
        "critical_rule": "challenge_split=test is never used for train/val/tuning, even when wmh.nii.gz exists locally",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {split_csv}, {heldout_csv}, {summary_json}")


if __name__ == "__main__":
    main()
