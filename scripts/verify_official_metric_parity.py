#!/usr/bin/env python3
"""Verify official metric parity report and claim boundaries (v2 schema)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify official metric parity report.")
    parser.add_argument("--report", default="reports/evaluation/official_metric_parity_report.json")
    parser.add_argument("--required-for-claim", default="", help="e.g. official_comparable")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / args.report
    if not report_path.exists():
        if args.required_for_claim:
            raise SystemExit(f"parity report missing for claim {args.required_for_claim}: {report_path}")
        print(f"parity report absent (optional): {report_path}")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if not report.get("official_evaluator_sha256"):
        failures.append("official_evaluator_sha256 missing")

    claims_allowed = report.get("claims_allowed", report.get("claim_allowed", {}))
    if args.required_for_claim == "official_comparable" and not claims_allowed.get("official_comparable"):
        failures.append("official_comparable claim not allowed by report")

    if claims_allowed.get("leaderboard_or_sota"):
        failures.append("blocked leaderboard_or_sota claim allowed")

    fixture_cases = report.get("fixture_results", report.get("fixture_cases", []))
    for case in fixture_cases:
        if not case.get("pass", False):
            case_id = case.get("fixture_id") or case.get("case_id") or "unknown"
            failures.append(f"fixture case failed: {case_id}")

    license_status = str(report.get("license_review_status", "")).upper()
    if args.required_for_claim and license_status not in {"APPROVED", "CONDITIONAL"}:
        failures.append(f"license_review_status not approved: {license_status}")

    if failures:
        raise SystemExit("official metric parity gate FAIL:\n" + "\n".join(failures))
    print(f"official metric parity gate PASS: {report_path}")


if __name__ == "__main__":
    main()
