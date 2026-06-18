#!/usr/bin/env python3
"""Verify official metric parity report and claim boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify official metric parity report.")
    parser.add_argument("--report", default="reports/official_metric_parity_report.json")
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

    claim_allowed = report.get("claim_allowed", {})
    if args.required_for_claim == "official_comparable" and not claim_allowed.get("official_comparable"):
        failures.append("official_comparable claim not allowed by report")

    for case in report.get("fixture_cases", []):
        if not case.get("pass", False):
            failures.append(f"fixture case failed: {case.get('case_id')}")

    license_status = str(report.get("license_review_status", "")).upper()
    if args.required_for_claim and license_status not in {"APPROVED", "CONDITIONAL"}:
        failures.append(f"license_review_status not approved: {license_status}")

    if failures:
        raise SystemExit("official metric parity gate FAIL:\n" + "\n".join(failures))
    print(f"official metric parity gate PASS: {report_path}")


if __name__ == "__main__":
    main()
