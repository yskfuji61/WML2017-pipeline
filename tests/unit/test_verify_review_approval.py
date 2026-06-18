import subprocess
import sys
from pathlib import Path


def test_verify_review_approval_fails_for_open_register():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify_review_approval.py"), "--target-state", "READY_FOR_PREVIEW"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "UNASSIGNED" in result.stderr or "UNASSIGNED" in result.stdout or "status != APPROVED" in result.stderr
