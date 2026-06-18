import json
import subprocess
import sys
from pathlib import Path


def test_verify_rollback_rehearsal_passes_for_preview():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_rollback_rehearsal.py"),
            "--target-state",
            "READY_FOR_PREVIEW",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_verify_rollback_rehearsal_requires_all_for_limited_use(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    rollback_dir = tmp_path / "rollback"
    rollback_dir.mkdir()
    (rollback_dir / "rollback_rehearsal_bad_config.json").write_text(
        json.dumps({"verification": {"status": "PASS"}, "rollback_target": {}, "commands": []}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_rollback_rehearsal.py"),
            "--target-state",
            "READY_FOR_LIMITED_INTERNAL_USE",
            "--rollback-dir",
            str(rollback_dir),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
