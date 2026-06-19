#!/usr/bin/env python3
"""v4 wrapper -> scripts/verify_package_identity.py"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "verify_package_identity.py"), run_name="__main__")
