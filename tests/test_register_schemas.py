"""Register schema smoke tests."""

import subprocess
import sys
from pathlib import Path


def test_validate_registers_passes():
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "scripts/validate_registers.py")], check=True, cwd=repo)
