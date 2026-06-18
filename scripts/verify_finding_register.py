#!/usr/bin/env python3
"""Verify finding register closure for target release state."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PREVIEW_STATES = {"READY_FOR_PREVIEW", "READY_FOR_LIMITED_INTERNAL_USE", "READY_FOR_RELEASE"}


def _is_open(status: str) -> bool:
    return status.upper().startswith("OPEN")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify finding register.")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--register", default="registry/finding_register_wmh2017.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    register_path = repo_root / args.register
    with register_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    failures: list[str] = []
    open_sev0_sev1: list[str] = []

    for row in rows:
        fid = row.get("finding_id", "")
        severity = row.get("severity", "")
        status = row.get("status", "")
        owner = row.get("owner", "")

        if owner.strip().upper().startswith("UNASSIGNED"):
            failures.append(f"{fid}: owner UNASSIGNED")

        if severity in {"Sev0", "Sev1"} and _is_open(status):
            open_sev0_sev1.append(fid)

        if status.upper() == "CLOSED":
            if not row.get("closure_hash", "").strip():
                failures.append(f"{fid}: CLOSED but closure_hash empty")
            if row.get("review_status", "").upper() != "APPROVED":
                failures.append(f"{fid}: CLOSED but review_status != APPROVED")
            evidence_path = row.get("evidence_path", "").strip()
            if evidence_path and not (repo_root / evidence_path).exists():
                failures.append(f"{fid}: evidence_path missing: {evidence_path}")

    if args.target_state in PREVIEW_STATES and open_sev0_sev1:
        failures.append(f"open Sev0/Sev1 findings: {open_sev0_sev1}")

    if failures:
        raise SystemExit("finding register gate FAIL:\n" + "\n".join(failures))
    print(f"finding register gate PASS for {args.target_state}")


if __name__ == "__main__":
    main()
