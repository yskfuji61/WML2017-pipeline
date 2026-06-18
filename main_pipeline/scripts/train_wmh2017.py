#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Enable before any torch import so PyTorch reads MPS fallback at backend init.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.training.train_monai import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WMH2017 MONAI smoke training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_main(args.config)
