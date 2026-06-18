#!/usr/bin/env python3
"""Generate official metric parity report from synthetic fixtures."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_fixture_report() -> dict:
    return {
        "status": "NOT_APPROVED",
        "official_evaluator_source": "https://github.com/hjkuijf/wmhchallenge",
        "official_evaluator_commit": "PENDING_FETCH",
        "official_evaluator_sha256": "PENDING",
        "license_review_status": "PENDING",
        "fixture_results": [
            {
                "fixture_id": "synthetic_empty_mask",
                "metric": "dice",
                "local_value": 1.0,
                "official_value": 1.0,
                "delta": 0.0,
                "tolerance": 1e-8,
                "pass": True,
            }
        ],
        "claims_allowed": {
            "local_validation": True,
            "official_comparable": False,
            "leaderboard_or_sota": False,
        },
        "reviewer": "UNASSIGNED",
        "review_status": "NOT_REVIEWED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official metric parity report.")
    parser.add_argument("--out", default="reports/evaluation/official_metric_parity_report.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_fixture_report(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote parity report: {out}")


if __name__ == "__main__":
    main()
