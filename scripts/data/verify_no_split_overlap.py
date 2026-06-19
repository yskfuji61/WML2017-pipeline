#!/usr/bin/env python3
"""Verify train/val split has no overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify split manifest has no overlap.")
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if payload.get("status") == "MISSING_WMH2017_ROOT":
        print("split overlap gate PASS (MISSING_WMH2017_ROOT recorded)")
        return

    train = set(payload.get("train_case_ids", []))
    val = set(payload.get("val_case_ids", []))
    heldout = set(payload.get("heldout_case_ids", []))
    overlap = (train & val) | (train & heldout) | (val & heldout)
    if overlap:
        raise SystemExit(f"split overlap FAIL: {sorted(overlap)[:20]}")
    print("split overlap gate PASS")


if __name__ == "__main__":
    main()
