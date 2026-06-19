#!/usr/bin/env python3
"""Fail on prohibited positive overclaims; allow negative-boundary wording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.security.overclaim import scan_tree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify no prohibited overclaim wording.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    hits = scan_tree(root)
    if hits:
        lines = [f"{hit.path}:{hit.line_no} [{hit.pattern_id}] {hit.line[:200]}" for hit in hits[:50]]
        raise SystemExit("overclaim gate FAIL:\n" + "\n".join(lines))
    print("overclaim gate PASS")


if __name__ == "__main__":
    main()
