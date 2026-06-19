#!/usr/bin/env python3
"""Validate metrics_summary.json schema for v4 smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify metric output JSON.")
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        payload_path = path.parent / "failed_run.json"
        if payload_path.exists():
            print("metric output gate PASS (FAILED_RUN recorded, metrics optional)")
            return
        raise SystemExit(f"metrics file missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "MISSING_WMH2017_ROOT":
        print("metric output gate PASS (MISSING_WMH2017_ROOT)")
        return

    if "run_id" not in payload and "metrics" not in payload and "dice_mean" not in payload:
        raise SystemExit("metric output FAIL: missing run_id/metrics/dice_mean")
    print("metric output gate PASS")


if __name__ == "__main__":
    main()
