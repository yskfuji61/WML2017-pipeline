#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.data.splits import (
    KFoldPolicy,
    assert_kfold_coverage,
    make_kfold_splits,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create site-stratified k-fold CV splits from challenge training cases only, "
            "with a test-contamination guard. Local validation only; not SOTA/clinical."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-col", default="site")
    parser.add_argument("--source-split-col", default="challenge_split")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_id = f"WMH2017-KFOLD-SEED{args.seed}"
    folds = make_kfold_splits(
        manifest,
        KFoldPolicy(
            split_id=split_id,
            k=args.k,
            seed=args.seed,
            stratify_col=args.stratify_col,
            source_split_col=args.source_split_col,
        ),
    )
    assert_kfold_coverage(folds)

    out_dir = Path(args.out_dir or f"data/splits/wmh2017_kfold_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_records = []
    for i, fold_df in enumerate(folds):
        fold_csv = out_dir / f"fold{i}.csv"
        fold_df.to_csv(fold_csv, index=False)
        tv = fold_df[fold_df["assigned_split"].isin(["train", "val"])]
        val = fold_df[fold_df["assigned_split"] == "val"]
        site_val_counts = val.groupby("site", dropna=False).size().reset_index(name="n").to_dict(orient="records")
        fold_records.append(
            {
                "fold": i,
                "fold_csv": str(fold_csv),
                "fold_csv_sha256": sha256_file(fold_csv),
                "n_train": int((fold_df["assigned_split"] == "train").sum()),
                "n_val": int(len(val)),
                "n_train_val_total": int(len(tv)),
                "val_site_counts": site_val_counts,
            }
        )

    summary = {
        "split_id": split_id,
        "k": args.k,
        "seed": args.seed,
        "stratify_col": args.stratify_col,
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256_file(Path(args.manifest)),
        "folds": fold_records,
        "critical_rule": (
            "challenge_split=test is never used for train/val/tuning across any fold; "
            "validation folds partition the training cases with no overlap"
        ),
        "claim_boundary": "local cross-validation only; not SOTA/official/clinical/production",
    }
    summary_json = out_dir / f"kfold_summary_seed{args.seed}.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.k} folds + {summary_json} under {out_dir}")


if __name__ == "__main__":
    main()
