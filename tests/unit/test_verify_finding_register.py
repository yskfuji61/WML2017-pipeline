import subprocess
import sys
from pathlib import Path


def test_finding_register_passes_for_structural_review():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_finding_register.py"),
            "--target-state",
            "READY_FOR_STRUCTURAL_REVIEW",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_finding_register_fails_for_preview_with_open_sev1():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_finding_register.py"),
            "--target-state",
            "READY_FOR_PREVIEW",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
