import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = REPO_ROOT / "scripts/fetch_official_evaluator.py"


def _run_fetch(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(FETCH_SCRIPT), *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def test_fetch_official_evaluator_requires_commit():
    result = _run_fetch(
        "--repo-url",
        "https://github.com/hjkuijf/wmhchallenge.git",
        "--expected-tree-sha256",
        "abc",
        "--source-record",
        "third_party/official_wmh_evaluator/SOURCE_RECORD.md",
        "--license-review",
        "APPROVED",
        "--output-dir",
        "third_party/official_wmh_evaluator/src",
    )
    assert result.returncode != 0


def test_fetch_official_evaluator_requires_license_approval():
    result = _run_fetch(
        "--repo-url",
        "https://github.com/hjkuijf/wmhchallenge.git",
        "--commit",
        "1" * 40,
        "--expected-tree-sha256",
        "a" * 64,
        "--source-record",
        "third_party/official_wmh_evaluator/SOURCE_RECORD.md",
        "--license-review",
        "NOT_REVIEWED",
        "--output-dir",
        "third_party/official_wmh_evaluator/src",
    )
    assert result.returncode != 0
    assert "APPROVED" in result.stderr + result.stdout


def test_fetch_official_evaluator_rejects_non_allowlisted_url():
    result = _run_fetch(
        "--repo-url",
        "https://example.com/evil.git",
        "--commit",
        "1" * 40,
        "--expected-tree-sha256",
        "a" * 64,
        "--source-record",
        "third_party/official_wmh_evaluator/SOURCE_RECORD.md",
        "--license-review",
        "APPROVED",
        "--output-dir",
        "third_party/official_wmh_evaluator/src",
    )
    assert result.returncode != 0
    assert "allowlist" in result.stderr + result.stdout


def test_verify_official_evaluator_source_fails_when_not_fetched():
    from scripts.verify_official_evaluator_source import verify_official_evaluator_source

    failures = verify_official_evaluator_source(REPO_ROOT, structure_only=True)
    assert failures
