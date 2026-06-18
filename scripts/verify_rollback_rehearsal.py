#!/usr/bin/env python3
"""Verify rollback rehearsal reports for target release state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIOS = [
    "bad_config",
    "wrong_threshold",
    "dependency_regression",
    "artifact_hash_mismatch",
    "split_contamination",
]

REQUIRE_ALL = {"READY_FOR_LIMITED_INTERNAL_USE", "READY_FOR_RELEASE"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify rollback rehearsal reports.")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--rollback-dir", default="reports/rollback")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rollback_dir = repo_root / args.rollback_dir
    failures: list[str] = []

    for scenario in SCENARIOS:
        path = rollback_dir / f"rollback_rehearsal_{scenario}.json"
        if not path.exists():
            if args.target_state in REQUIRE_ALL:
                failures.append(f"missing rollback report: {path.name}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        verification = data.get("verification", {})
        status = verification.get("status") or data.get("verification_result")
        if status != "PASS":
            failures.append(f"{scenario}: verification status != PASS")
        if args.target_state in REQUIRE_ALL:
            if not data.get("rollback_target"):
                failures.append(f"{scenario}: missing rollback_target")
            if not data.get("commands"):
                failures.append(f"{scenario}: missing commands")

    if failures:
        raise SystemExit("rollback rehearsal gate FAIL:\n" + "\n".join(failures))
    print(f"rollback rehearsal gate PASS for {args.target_state}")


if __name__ == "__main__":
    main()
