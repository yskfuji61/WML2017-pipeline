#!/usr/bin/env python3
"""Verify metric formula lock doc lists required v4 metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED = (
    "dice_local",
    "hd95_local_mm",
    "avd_local_percent",
    "lavd_wmh2017_compat_candidate",
    "lesion_recall_local",
    "lesion_f1_local",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify metric formula lock spec.")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    text = Path(args.spec).read_text(encoding="utf-8")
    missing = [metric for metric in REQUIRED if metric not in text]
    if missing:
        raise SystemExit(f"metric formula lock FAIL missing: {missing}")
    if not re.search(r"prediction_manifest_hash", text):
        raise SystemExit("metric formula lock FAIL: prediction_manifest_hash not documented")
    print("metric formula lock gate PASS")


if __name__ == "__main__":
    main()
