#!/usr/bin/env python3
"""Generate visual QA report manifest (no PNG commit by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual report figure manifest.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": args.run_id,
        "selection": {"best": "PENDING", "median": "PENDING", "worst": "PENDING"},
        "figures": [],
        "png_committed_by_default": False,
        "local_path_redacted": "REDACTED_OR_LOCAL_ONLY",
        "public_data_only": True,
    }
    manifest_path = out_dir / f"{args.run_id}_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
