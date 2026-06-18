from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_FILES = (
    "structural_ci.yml",
    "security_scan.yml",
    "release_candidate_ci.yml",
)


def test_github_actions_workflow_is_at_repository_root():
    assert WORKFLOWS_DIR.is_dir()
    assert not (REPO_ROOT / "main_pipeline" / ".github" / "workflows").exists()

    for name in WORKFLOW_FILES:
        workflow_path = WORKFLOWS_DIR / name
        assert workflow_path.is_file(), f"missing workflow: {workflow_path}"

        content = workflow_path.read_text(encoding="utf-8")
        assert "permissions:" in content
        assert "contents: read" in content

    structural = (WORKFLOWS_DIR / "structural_ci.yml").read_text(encoding="utf-8")
    assert "working-directory: ." in structural
