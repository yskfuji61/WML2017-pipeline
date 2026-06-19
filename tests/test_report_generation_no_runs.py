"""Biweekly report works with zero runs."""

import subprocess
import sys
from pathlib import Path


def test_generate_biweekly_report_zero_runs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/reporting/generate_biweekly_report.py"),
            "--period-start",
            "2026-06-16",
            "--period-end",
            "2026-06-30",
            "--output",
            str(out),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        cwd=repo,
    )
    text = out.read_text(encoding="utf-8")
    assert "NO_REAL_RUN_EVIDENCE" in text
