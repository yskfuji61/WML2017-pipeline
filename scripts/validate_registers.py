#!/usr/bin/env python3
"""Validate register CSV schemas (v4 skeleton)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED = {
    "registry/agent_run_register.csv": {"agent_run_id", "pr_phase", "status"},
    "registry/claim_register_wmh2017.csv": {"claim_id", "claim_text", "status"},
    "registry/dataset_register_wmh2017.csv": {"dataset_register_id", "dataset_id", "status"},
    "registry/split_register_wmh2017.csv": {"split_register_id", "split_id", "status"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate register schemas.")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root)
    failures: list[str] = []
    for rel, required_cols in REQUIRED.items():
        path = root / rel
        if not path.exists():
            failures.append(f"missing register: {rel}")
            continue
        with path.open(encoding="utf-8", newline="") as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        missing = required_cols - cols
        if missing:
            failures.append(f"{rel} missing columns: {sorted(missing)}")
    if failures:
        raise SystemExit("register validation FAIL:\n" + "\n".join(failures))
    print("register validation PASS")


if __name__ == "__main__":
    main()
