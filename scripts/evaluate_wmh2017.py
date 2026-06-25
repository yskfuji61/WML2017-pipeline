#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.evaluate_predictions import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WMH2017 validation predictions with local metrics. Does not create official/SOTA claims."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--assigned-split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-artifact", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--allow-shape-only-geometry", action="store_true")
    parser.add_argument(
        "--skip-missing-predictions",
        action="store_true",
        help="Evaluate only cases with prediction files (smoke runs with val_max_cases)",
    )
    parser.add_argument(
        "--allow-released-label-local-test",
        action="store_true",
        help=(
            "Default-off override permitting challenge_split=test cases ONLY for "
            "assigned_split=heldout_eval (released-label LOCAL test; not official/leaderboard/SOTA)."
        ),
    )
    args = parser.parse_args()

    summary = evaluate_predictions(
        manifest_csv=args.manifest,
        split_csv=args.split,
        prediction_dir=args.predictions,
        out_dir=args.out_dir,
        run_id=args.run_id,
        assigned_split=args.assigned_split,
        threshold=args.threshold,
        strict_geometry=not args.allow_shape_only_geometry,
        model_artifact_path=args.model_artifact,
        config_path=args.config_path,
        skip_missing_predictions=args.skip_missing_predictions,
        allow_released_label_local_test=args.allow_released_label_local_test,
    )
    print(f"Wrote {summary['case_metrics_csv']}")
    print(f"Wrote {Path(args.out_dir) / 'metrics_summary.json'}")
    print(f"mean_dice={summary['mean_dice']:.6f} n_cases={summary['n_cases']}")


if __name__ == "__main__":
    main()
