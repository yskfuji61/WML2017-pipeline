import subprocess
import sys
from pathlib import Path


def test_verify_package_identity_passes():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify_package_identity.py")],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "0.2.3" in result.stdout
