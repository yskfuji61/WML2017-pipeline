from pathlib import Path


def test_verify_review_approval_fails_on_unassigned(tmp_path: Path, monkeypatch):
    repo = tmp_path
    (repo / "registry").mkdir()
    (repo / "registry" / "review_approval_register_wmh2017.csv").write_text(
        "record_id,record_type,artifact_or_scope,version_or_hash,reviewer,role_qualification,"
        "independence_required,review_scope,date,status,comments,disposition,conditions,"
        "linked_finding_ids,approval_meaning,release_impact\n"
        "REV-WMH-001,source_review,scope,PENDING,UNASSIGNED_HUMAN_REVIEWER,r,y,scope,date,OPEN,,NOT_APPROVED,,,,\n",
        encoding="utf-8",
    )
    (repo / "registry" / "finding_register_wmh2017.csv").write_text(
        "finding_id,severity,status,title,evidence_absent_or_gap,affected_claims,release_impact,"
        "owner,reviewer,required_fix,closure_evidence,due_or_review_trigger\n",
        encoding="utf-8",
    )
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "verify_review_approval.py"),
            "--target-state",
            "READY_FOR_PREVIEW",
            "--review-register",
            str(repo / "registry" / "review_approval_register_wmh2017.csv"),
            "--finding-register",
            str(repo / "registry" / "finding_register_wmh2017.csv"),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_rollback_rehearsal_all_scenarios():
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_rollback_rehearsal.py"), "--all-scenarios"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
