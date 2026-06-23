#!/usr/bin/env python3
"""Evaluate a frozen N-member ensemble spec on validation probability maps.

Weights/threshold come from the spec and are never tuned here. Never consumes the test
split; writes a hashed evaluation artifact recording the frozen spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.ensemble import (
    evaluate_ensemble,
    spec_from_config,
    write_ensemble_evaluation_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ensemble spec on val (no weight tuning).")
    parser.add_argument("--spec-config", required=True, help="YAML with an ensemble block")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--assigned-split", default="val")
    args = parser.parse_args()

    spec = spec_from_config(yaml.safe_load(Path(args.spec_config).read_text(encoding="utf-8")))
    summary = evaluate_ensemble(
        spec=spec,
        manifest_csv=args.manifest,
        split_csv=args.split,
        assigned_split=args.assigned_split,
    )
    payload = write_ensemble_evaluation_artifact(out_dir=args.out_dir, summary=summary, spec=spec, run_id=args.run_id)

    print(f"Wrote {Path(args.out_dir) / 'ensemble_evaluation.json'}")
    print(
        f"members={payload['member_names']} weights={payload['weights']} "
        f"mean_dice={payload['mean_dice']:.6f} mean_lesion_recall={payload['mean_lesion_recall']:.6f} "
        f"n_cases={payload['n_cases']}"
    )


if __name__ == "__main__":
    main()
