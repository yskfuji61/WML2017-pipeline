#!/usr/bin/env python3
"""v4 wrapper -> scripts/generate_sbom.py"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "generate_sbom.py"), run_name="__main__")
