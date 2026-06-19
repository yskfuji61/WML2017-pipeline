#!/usr/bin/env python3
"""v4 wrapper: create train/val split manifest JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val split manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if manifest.suffix == ".json":
        csv_candidate = manifest.with_suffix(".csv")
        if not csv_candidate.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("status") == "MISSING_WMH2017_ROOT":
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps({"status": "MISSING_WMH2017_ROOT", "split_hash": "MISSING_WMH2017_ROOT"}, indent=2),
                    encoding="utf-8",
                )
                print("split manifest recorded MISSING_WMH2017_ROOT")
                return
            raise SystemExit(f"CSV companion missing for JSON manifest: {csv_candidate}")
        manifest = csv_candidate

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "splits"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_wmh2017_splits.py"),
            "--manifest",
            str(manifest),
            "--seed",
            str(args.seed),
            "--out-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)
        split_csv = out_dir / f"wmh2017_train_val_seed{args.seed}.csv"
        import pandas as pd

        df = pd.read_csv(split_csv)
        split_payload = {
            "split_id": f"WMH2017-TRAIN-VAL-SEED{args.seed}",
            "seed": args.seed,
            "train_case_ids": df[df["assigned_split"] == "train"]["case_id"].astype(str).tolist(),
            "val_case_ids": df[df["assigned_split"] == "val"]["case_id"].astype(str).tolist(),
            "heldout_case_ids": [],
        }
        split_payload["split_hash"] = hashlib.sha256(
            json.dumps(split_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
