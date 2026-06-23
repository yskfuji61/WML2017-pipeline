#!/usr/bin/env python3
"""Audit lesion recall by GT size bin: baseline vs component-filtered (val only).

Surfaces small-lesion deletion caused by a connected-component min-size filter, so a
Dice-improving postprocess that hurts small-lesion recall is detected. Never tunes on the
test split; does not change the selected checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.threshold_sweep import (
    lesion_size_bin_audit,
    write_lesion_size_bin_audit_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesion-size-bin recall audit (val-only postprocess diagnostic).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--probs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-component-size", type=int, required=True)
    parser.add_argument("--assigned-split", default="val")
    args = parser.parse_args()

    audit = lesion_size_bin_audit(
        manifest_csv=args.manifest,
        split_csv=args.split,
        probs_dir=args.probs_dir,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
        assigned_split=args.assigned_split,
    )
    payload = write_lesion_size_bin_audit_artifact(out_dir=args.out_dir, audit=audit, run_id=args.run_id)

    print(f"Wrote {Path(args.out_dir) / 'lesion_size_bin_audit.json'}")
    for row in payload["per_bin"]:
        print(
            f"  bin={row['bin']:<7} n_target={row['n_target']:<4} "
            f"baseline_recall={row['baseline_recall']} -> post_recall={row['post_recall']} "
            f"(delta={row['delta_recall']})"
        )
    print(f"small_lesion_recall_regressed={payload['small_lesion_recall_regressed']}")


if __name__ == "__main__":
    main()
