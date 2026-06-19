#!/usr/bin/env python3
"""Verify review approval register for target release state (v2 schema)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PREVIEW_STATES = {"READY_FOR_PREVIEW", "READY_FOR_LIMITED_INTERNAL_USE", "READY_FOR_RELEASE"}

REQUIRED_REVIEWS = {
    "READY_FOR_PREVIEW": {
        "source_review",
        "model_validation_review",
        "security_privacy_review",
        "release_approval",
    },
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify review approval register.")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--review-register", default="registry/review_approval_register_wmh2017.csv")
    parser.add_argument("--finding-register", default="registry/finding_register_wmh2017.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reviews = _load_csv(repo_root / args.review_register)
    findings = _load_csv(repo_root / args.finding_register)
    failures: list[str] = []

    if args.target_state not in PREVIEW_STATES:
        print(f"review approval gate SKIP for {args.target_state}")
        return

    required_types = REQUIRED_REVIEWS.get(args.target_state, set())
    seen_types = {r.get("review_type", "") for r in reviews}
    for req in required_types:
        if req not in seen_types:
            failures.append(f"missing review_type: {req}")

    open_sev1 = {
        f["finding_id"]
        for f in findings
        if f.get("severity", "").startswith("Sev1") and f.get("status", "").upper().startswith("OPEN")
    }

    for row in reviews:
        rtype = row.get("review_type", "")
        if rtype not in required_types:
            continue
        rid = row.get("review_id", row.get("record_id", ""))
        if "UNASSIGNED" in row.get("reviewer", ""):
            failures.append(f"{rid}: reviewer UNASSIGNED")
        if row.get("status", "").upper() != "APPROVED":
            failures.append(f"{rid}: status != APPROVED ({row.get('status')})")
        artifact_hash = row.get("artifact_hash", row.get("version_or_hash", ""))
        if not artifact_hash.strip() or "PENDING" in artifact_hash:
            failures.append(f"{rid}: artifact_hash pending")
        decision = row.get("decision", row.get("disposition", ""))
        if not decision.strip() or decision.upper() == "NOT_APPROVED":
            failures.append(f"{rid}: decision not approved")
        review_date = row.get("review_date", row.get("date", ""))
        if not review_date.strip() or review_date == "NOT_REVIEWED":
            failures.append(f"{rid}: review_date empty")
        linked = row.get("linked_findings", row.get("linked_finding_ids", ""))
        for fid in linked.replace(";", " ").split():
            fid = fid.strip()
            if fid in open_sev1:
                failures.append(f"{rid}: linked Sev1 open: {fid}")

    if failures:
        raise SystemExit("review approval gate FAIL:\n" + "\n".join(failures))
    print(f"review approval gate PASS for {args.target_state}")


if __name__ == "__main__":
    main()
