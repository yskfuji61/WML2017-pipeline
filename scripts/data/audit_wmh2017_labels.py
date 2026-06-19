#!/usr/bin/env python3
"""v4 wrapper -> scripts/audit_wmh2017_labels.py"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "audit_wmh2017_labels.py"), run_name="__main__")
