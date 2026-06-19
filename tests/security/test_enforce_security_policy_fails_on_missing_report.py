import json
from pathlib import Path

from scripts.enforce_security_policy import enforce_security_policy


def test_enforce_security_policy_fails_on_missing_report(tmp_path: Path):
    report_dir = tmp_path / "reports/security"
    report_dir.mkdir(parents=True)
    repo_root = tmp_path
    (repo_root / "registry").mkdir()
    sbom = report_dir / "sbom.cdx.json"
    sbom.write_text(json.dumps({"components": [{"name": "pkg"}]}), encoding="utf-8")

    _, failures = enforce_security_policy(
        report_dir,
        repo_root=repo_root,
        exception_register=repo_root / "registry/security_exception_register.csv",
        sbom_path=sbom,
    )
    assert failures
    assert any("missing" in item.lower() for item in failures)
