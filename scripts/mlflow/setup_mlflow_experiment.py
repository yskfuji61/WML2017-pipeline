#!/usr/bin/env python3
"""Setup local-only MLflow experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup local MLflow experiment.")
    parser.add_argument("--config", default="configs/mlflow/local_mlflow.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    uri = cfg.get("tracking_uri", "file:./mlruns")
    if uri.startswith("http"):
        raise SystemExit("remote MLflow refused by default")
    print(f"MLflow local tracking OK: {uri}")


if __name__ == "__main__":
    main()
