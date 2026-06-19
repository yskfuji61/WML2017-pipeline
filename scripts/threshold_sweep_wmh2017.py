#!/usr/bin/env python3
"""Run validation-only threshold sweep over saved probability maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.threshold_sweep import (
    select_best_threshold,
    sweep_thresholds,
    write_binary_predictions_at_threshold,
    write_threshold_sweep_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep validation thresholds on saved probability maps (val only).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--probs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--assigned-split", default="val")
    parser.add_argument("--training-threshold", type=float, default=0.5)
    parser.add_argument("--write-predictions-dir", default="")
    args = parser.parse_args()

    summary_df, per_case_df = sweep_thresholds(
        manifest_csv=args.manifest,
        split_csv=args.split,
        probs_dir=args.probs_dir,
        assigned_split=args.assigned_split,
    )
    best = select_best_threshold(summary_df)
    payload = write_threshold_sweep_artifacts(
        out_dir=args.out_dir,
        summary_df=summary_df,
        per_case_df=per_case_df,
        best=best,
        run_id=args.run_id,
        probs_dir=args.probs_dir,
        training_threshold=args.training_threshold,
    )
    print(f"Wrote {payload['summary_csv']}")
    print(f"Wrote {Path(args.out_dir) / 'threshold_sweep_best.json'}")
    print(
        "best_threshold="
        f"{best['threshold']:.4f} mean_dice={best['mean_dice']:.6f} "
        f"mean_lesion_recall={best['mean_lesion_recall']:.6f}"
    )
    if args.write_predictions_dir:
        n_written = write_binary_predictions_at_threshold(
            manifest_csv=args.manifest,
            split_csv=args.split,
            probs_dir=args.probs_dir,
            prediction_dir=args.write_predictions_dir,
            threshold=float(best["threshold"]),
            assigned_split=args.assigned_split,
        )
        print(f"Wrote {n_written} thresholded predictions to {args.write_predictions_dir}")


if __name__ == "__main__":
    main()
