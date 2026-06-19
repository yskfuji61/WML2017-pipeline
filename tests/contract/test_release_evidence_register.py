from pathlib import Path

from scripts.verify_release_evidence_register import REQUIRED_COLUMNS, verify_register


def test_release_evidence_register_has_required_columns():
    repo_root = Path(__file__).resolve().parents[2]
    register = repo_root / "registry/release_evidence_register_wmh2017.csv"
    text = register.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    assert header == REQUIRED_COLUMNS


def test_release_evidence_register_structure_passes():
    repo_root = Path(__file__).resolve().parents[2]
    failures = verify_register(repo_root, run_id="wmh2017_preview_20260618_e48ed25", structure_only=True)
    assert failures == []


def test_release_evidence_register_rejects_duplicate_ids(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "registry").mkdir(parents=True)
    register = repo_root / "registry/release_evidence_register_wmh2017.csv"
    register.write_text(
        "evidence_id,run_id,commit_sha,ci_run_url,artifact_path,artifact_sha256,evidence_type,reviewer,decision,created_at_utc\n"
        "RE-WMH-001,run1,1d34fd902fe817072e971fcad01fd9695a64d7c9,PENDING,docs/release_evidence/README.md,PENDING,test,,pass,2026-06-19T00:00:00+00:00\n"
        "RE-WMH-001,run1,1d34fd902fe817072e971fcad01fd9695a64d7c9,PENDING,docs/release_evidence/README.md,PENDING,test,,pass,2026-06-19T00:00:00+00:00\n",
        encoding="utf-8",
    )
    (repo_root / "src/wmh2017/registry/schemas").mkdir(parents=True)
    failures = verify_register(repo_root, structure_only=True)
    assert any("duplicate evidence_id" in item for item in failures)
