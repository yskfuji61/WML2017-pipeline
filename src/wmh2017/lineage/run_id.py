"""Canonical run ID generation for WMH2017 preview runs."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone

RUN_ID_PATTERN = re.compile(r"^wmh2017_preview_\d{8}_[0-9a-f]{7,40}$")


def git_short_sha() -> str:
    try:
        full = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        return full[:7] if full else "unknown"
    except Exception:
        return "unknown"


def build_preview_run_id(*, date: datetime | None = None) -> str:
    dt = date or datetime.now(timezone.utc)
    return f"wmh2017_preview_{dt.strftime('%Y%m%d')}_{git_short_sha()}"


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"run_id must match wmh2017_preview_YYYYMMDD_gitsha, got: {run_id}"
        )


def assert_run_dir_unique(run_dir_exists: bool, run_id: str) -> None:
    if run_dir_exists:
        raise FileExistsError(f"run_id already exists (refuse overwrite): {run_id}")
