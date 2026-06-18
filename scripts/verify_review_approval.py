#!/usr/bin/env python3
"""Verify review approval register for target release state."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
    parser.add_argument("--review-register", default="registry/review_approval_register_wmh2017.csv")
    parser.add_argument("--finding-register", default="registry/finding_register_wmh2017.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reviews = _load_csv(repo_root / args.review_register)
    findings = _load_csv(repo_root / args.finding_register)
    failures: list[str] = []

    required_types = REQUIRED_REVIEWS.get(args.target_state, set())
    seen_types = {r.get("record_type", "") for r in reviews}
    for req in required_types:
        if req not in seen_types:
            failures.append(f"missing review record_type: {req}")

    open_sev1 = {
        f["finding_id"]
        for f in findings
        if f.get("severity", "").startswith("Sev1") and f.get("status", "").upper().startswith("OPEN")
    }

    for row in reviews:
        rtype = row.get("record_type", "")
        if required_types and rtype not in required_types:
            continue
        if "UNASSIGNED" in row.get("reviewer", ""):
            failures.append(f"{row.get('record_id')}: reviewer UNASSIGNED")
        if row.get("status", "").upper() != "APPROVED":
            failures.append(f"{row.get('record_id')}: status != APPROVED ({row.get('status')})")
        if "PENDING" in row.get("version_or_hash", ""):
            failures.append(f"{row.get('record_id')}: version_or_hash pending")
        linked = row.get("linked_finding_ids", "")
        for fid in linked.replace(";", " ").split():
            fid = fid.strip()
            if fid in open_sev1:
                failures.append(f"{row.get('record_id')}: linked Sev1 open: {fid}")

    if failures:
        raise SystemExit("review approval gate FAIL:\n" + "\n".join(failures))
    print(f"review approval gate PASS for {args.target_state}")


if __name__ == "__main__":
    main()
