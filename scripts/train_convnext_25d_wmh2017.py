#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.training.train_convnext_25d import train_convnext_25d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WMH2017 2.5D ConvNeXt-Tiny model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = train_convnext_25d(args.config)
    print(result)
