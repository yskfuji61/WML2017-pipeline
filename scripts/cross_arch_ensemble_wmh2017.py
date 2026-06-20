#!/usr/bin/env python3
"""Cross-architecture ensemble evaluation CLI for WMH2017 val-only tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.cross_arch_ensemble import evaluate_fused_predictions, sweep_ensemble_hyperparams


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse two probability dirs and evaluate on val.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--primary-probs-dir", required=True)
    parser.add_argument("--secondary-probs-dir", required=True)
    parser.add_argument("--assigned-split", default="val")
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    if args.sweep:
        df = sweep_ensemble_hyperparams(
            manifest_csv=args.manifest,
            split_csv=args.split,
            primary_probs_dir=args.primary_probs_dir,
            secondary_probs_dir=args.secondary_probs_dir,
            assigned_split=args.assigned_split,
        )
        if args.out_csv:
            out = Path(args.out_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Wrote {out}")
        best = df.iloc[0]
        print(
            f"best secondary_weight={best['secondary_weight']:.3f} "
            f"threshold={best['threshold']:.3f} mean_dice={best['mean_dice']:.6f}"
        )
        return

    summary = evaluate_fused_predictions(
        manifest_csv=args.manifest,
        split_csv=args.split,
        primary_probs_dir=args.primary_probs_dir,
        secondary_probs_dir=args.secondary_probs_dir,
        assigned_split=args.assigned_split,
        secondary_weight=args.secondary_weight,
        threshold=args.threshold,
    )
    print(
        f"mean_dice={summary['mean_dice']:.6f} "
        f"mean_lesion_recall={summary['mean_lesion_recall']:.6f} "
        f"n_cases={summary['n_cases']}"
    )


if __name__ == "__main__":
    main()
