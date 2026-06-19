#!/usr/bin/env python3
"""Verify manager report claims against v4 policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.claim_policy import scan_report_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify manager report claim wording.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    text = Path(args.report).read_text(encoding="utf-8")
    hits = scan_report_text(text)
    if hits:
        raise SystemExit("manager report claim gate FAIL:\n" + "\n".join(hits[:30]))
    print("manager report claim gate PASS")


if __name__ == "__main__":
    main()
