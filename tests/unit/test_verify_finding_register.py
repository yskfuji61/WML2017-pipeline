import csv
import subprocess
import sys
from pathlib import Path


def _ensure_closed_run_evidence(repo: Path) -> None:
    with (repo / "registry/finding_register_wmh2017.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["finding_id"] == "FIND-WMH-003")
    evidence = repo / row["evidence_path"]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    if not evidence.exists():
        evidence.write_text("{}\n", encoding="utf-8")


def test_finding_register_passes_for_structural_review():
    repo = Path(__file__).resolve().parents[2]
    _ensure_closed_run_evidence(repo)
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


def test_finding_register_passes_for_preview_after_sev1_closure():
    repo = Path(__file__).resolve().parents[2]
    _ensure_closed_run_evidence(repo)
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
    assert result.returncode == 0, result.stderr or result.stdout
