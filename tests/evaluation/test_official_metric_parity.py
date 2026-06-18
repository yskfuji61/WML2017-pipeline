import json
import subprocess
import sys
from pathlib import Path


def test_parity_optional_when_report_missing():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify_official_metric_parity.py")],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_parity_fixture_report_passes(tmp_path: Path):
    report = {
        "official_evaluator_source": "https://github.com/hjkuijf/wmhchallenge",
        "official_evaluator_commit": "fixture",
        "official_evaluator_sha256": "sha256:fixture",
        "license_review_status": "CONDITIONAL",
        "fixture_cases": [
            {"case_id": "synthetic_empty", "local_dice": 1.0, "official_dice": 1.0, "delta": 0.0, "pass": True}
        ],
        "claim_allowed": {"local_validation": True, "official_comparable": False, "leaderboard_or_sota": False},
    }
    report_path = tmp_path / "official_metric_parity_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_official_metric_parity.py"),
            "--report",
            str(report_path),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
