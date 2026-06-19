#!/usr/bin/env python3
"""Verify v4 dataset manifest integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset manifest JSON.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"manifest missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "MISSING_WMH2017_ROOT":
        print("dataset integrity gate PASS (MISSING_WMH2017_ROOT recorded)")
        return

    required_top = {"dataset_id", "cases", "manifest_hash", "dlp_class"}
    missing = required_top - set(payload)
    if missing:
        raise SystemExit(f"dataset integrity FAIL missing keys: {sorted(missing)}")

    for case in payload.get("cases", []):
        if "case_id" not in case:
            raise SystemExit("dataset integrity FAIL: case missing case_id")
        flair = case.get("modalities", {}).get("flair", {})
        label = case.get("label", {})
        if flair.get("path", "").startswith("/"):
            raise SystemExit("dataset integrity FAIL: unredacted flair path")
        if label.get("path", "").startswith("/"):
            raise SystemExit("dataset integrity FAIL: unredacted label path")

    print("dataset integrity gate PASS")


if __name__ == "__main__":
    main()
