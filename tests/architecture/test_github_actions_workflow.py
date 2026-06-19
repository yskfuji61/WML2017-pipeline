import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_FILES = (
    "structural_ci.yml",
    "security_scan.yml",
    "release_candidate_ci.yml",
    "evidence_binder_ci.yml",
    "dependency_review.yml",
    "license_scan.yml",
)

TAG_ONLY_ACTION_REF = re.compile(r"uses:\s*(?:[\w./-]+/)?[\w./-]+@v\d+(?:\.\d+)*\s*(?:#.*)?$")


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
    assert "ruff format --check" in structural


def test_github_actions_use_commit_sha_pins():
    for name in WORKFLOW_FILES:
        content = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
        for line in content.splitlines():
            if not line.strip().startswith("- uses:") and not line.strip().startswith("uses:"):
                continue
            if "uses:" not in line:
                continue
            assert not TAG_ONLY_ACTION_REF.match(
                line.strip()
            ), f"{name} uses tag-only action ref (must pin full commit SHA): {line.strip()}"
            if line.strip().startswith(("- uses:", "uses:")):
                assert re.search(r"@[0-9a-f]{40}\b", line), f"{name} action ref missing 40-char SHA: {line.strip()}"
