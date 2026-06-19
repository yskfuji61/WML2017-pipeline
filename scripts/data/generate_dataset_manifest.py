#!/usr/bin/env python3
"""v4 wrapper: generate dataset manifest JSON from WMH2017 root."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from wmh2017.data.manifest import build_manifest
from wmh2017.schemas import manifest_json_from_csv, write_manifest_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v4 dataset manifest JSON.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inspect-images", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        payload = {
            "dataset_id": "wmh2017_public_poc",
            "status": "MISSING_WMH2017_ROOT",
            "root": "REDACTED_OR_LOCAL_ONLY",
            "cases": [],
            "manifest_hash": "MISSING_WMH2017_ROOT",
        }
        write_manifest_json(payload, Path(args.output))
        print(f"Recorded MISSING_WMH2017_ROOT at {args.output}", file=sys.stderr)
        return

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "dataset_manifest.csv"
        df = build_manifest(root, inspect_images=args.inspect_images)
        if df.empty:
            raise SystemExit("No WMH2017 cases found under root.")
        df.to_csv(csv_path, index=False)
        payload = manifest_json_from_csv(df, root=str(root))
        write_manifest_json(payload, Path(args.output))
        csv_out = Path(args.output).with_suffix(".csv")
        df.to_csv(csv_out, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
