#!/usr/bin/env python3
"""v4 wrapper -> scripts/run_rollback_rehearsal.py (reproducibility rehearsal, not production rollback)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_rollback_rehearsal.py"), run_name="__main__")
