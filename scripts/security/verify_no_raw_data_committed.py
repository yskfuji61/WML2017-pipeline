#!/usr/bin/env python3
"""v4 path wrapper -> scripts/verify_no_raw_data_committed.py"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "verify_no_raw_data_committed.py"
    runpy.run_path(str(target), run_name="__main__")
