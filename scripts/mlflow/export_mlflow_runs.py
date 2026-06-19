#!/usr/bin/env python3
"""Export MLflow runs index (zero runs OK)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLflow runs to CSV.")
    parser.add_argument("--config", default="configs/mlflow/local_mlflow.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if str(cfg.get("tracking_uri", "")).startswith("http"):
        raise SystemExit("remote MLflow refused")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "experiment_name", "status"])
    print(f"Wrote empty experiment index {out}")


if __name__ == "__main__":
    main()
