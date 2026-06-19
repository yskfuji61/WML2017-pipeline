#!/usr/bin/env python3
"""Backfill threshold sweep metadata into an existing run_evidence.json (local only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def backfill_threshold_policy(
    *,
    run_dir: str | Path,
    sweep_best_path: str | Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    evidence_path = run_dir / "run_evidence.json"
    if not evidence_path.exists():
        raise FileNotFoundError(f"run_evidence not found: {evidence_path}")

    sweep_path = (
        Path(sweep_best_path) if sweep_best_path else run_dir / "evaluation/threshold_sweep/threshold_sweep_best.json"
    )
    if not sweep_path.exists():
        raise FileNotFoundError(f"threshold sweep artifact not found: {sweep_path}")

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    policy = sweep.get("threshold_policy") or {}
    best = sweep.get("best") or {}

    evidence["threshold_policy"] = {
        "training_threshold": float(
            policy.get("training_threshold", evidence.get("threshold_policy", {}).get("training_threshold", 0.5))
        ),
        "sweep_best_threshold": float(policy.get("sweep_best_threshold", best.get("threshold", 0.5))),
        "sweep_split": str(policy.get("sweep_split", "val")),
        "selection_policy": str(
            policy.get("selection_policy", "max mean_dice on val; tie-break lesion_recall then lesion_f1")
        ),
    }
    evidence["threshold_sweep_artifact"] = str(sweep_path)
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence["threshold_policy"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill threshold_policy into run_evidence.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sweep-best", default="")
    args = parser.parse_args()
    policy = backfill_threshold_policy(
        run_dir=args.run_dir,
        sweep_best_path=args.sweep_best or None,
    )
    print(f"Updated threshold_policy: {policy}")


if __name__ == "__main__":
    main()
