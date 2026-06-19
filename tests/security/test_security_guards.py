"""Security guard tests (v4 PR-0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def test_raw_data_guard_passes():
    result = _run("verify_no_raw_data_committed.py", ".")
    assert result.returncode == 0, result.stderr or result.stdout


def test_overclaim_guard_passes_on_repo():
    result = _run("verify_no_overclaim.py", ".")
    assert result.returncode == 0, result.stderr or result.stdout


def test_env_secret_guard_passes():
    result = _run("verify_no_env_secrets.py", ".")
    assert result.returncode == 0, result.stderr or result.stdout


def test_remote_uri_guard_passes():
    result = _run("verify_no_unapproved_remote_uri.py", ".")
    assert result.returncode == 0, result.stderr or result.stdout


def test_security_wrappers_delegate():
    for name in (
        "verify_no_raw_data_committed.py",
        "verify_no_overclaim.py",
        "verify_no_absolute_paths.py",
        "verify_no_env_secrets.py",
        "verify_no_unapproved_remote_uri.py",
    ):
        wrapper = REPO_ROOT / "scripts" / "security" / name
        assert wrapper.is_file()
        result = subprocess.run([sys.executable, str(wrapper), "."], cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"{name}: {result.stderr or result.stdout}"
