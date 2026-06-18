#!/usr/bin/env python3
"""Fetch official WMH evaluator source under reviewed process."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

DEFAULT_URL = "https://github.com/hjkuijf/wmhchallenge.git"
TARGET = Path("third_party/official_wmh_evaluator/src")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official WMH evaluator (requires license review).")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--commit", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / TARGET
    if args.dry_run:
        print(f"would clone {args.url} into {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"target already exists: {target}")
    subprocess.check_call(["git", "clone", "--depth", "1", args.url, str(target)], cwd=str(repo_root))
    if args.commit:
        subprocess.check_call(["git", "checkout", args.commit], cwd=str(target))

    digest = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        if path.is_file():
            digest.update(path.read_bytes())
    sha_path = repo_root / "third_party/official_wmh_evaluator/evaluator.sha256"
    sha_path.write_text(digest.hexdigest() + "\n", encoding="utf-8")
    print(f"Wrote evaluator hash: {sha_path}")


if __name__ == "__main__":
    main()
