import json
from pathlib import Path

from scripts.enforce_security_policy import check_bandit, check_pip_audit, check_sbom


def test_bandit_high_finding_fails() -> None:
    report = {
        "results": [
            {
                "issue_severity": "HIGH",
                "issue_confidence": "HIGH",
                "test_id": "B101",
                "filename": "src/wmh2017/example.py",
                "line_number": 1,
            }
        ]
    }
    status, failures, _ = check_bandit(report, [])
    assert status == "FAIL"
    assert failures


def test_pip_audit_critical_fails() -> None:
    report = [{"name": "badpkg", "vulns": [{"id": "CVE-TEST-001", "severity": "CRITICAL"}]}]
    status, failures, _ = check_pip_audit(report, [])
    assert status == "FAIL"
    assert failures


def test_sbom_missing_fails(tmp_path: Path) -> None:
    status, failures, _ = check_sbom(tmp_path / "missing.cdx.json")
    assert status == "FAIL"
    assert failures


def test_sbom_nonempty_passes(tmp_path: Path) -> None:
    path = tmp_path / "sbom.cdx.json"
    path.write_text(json.dumps({"components": [{"name": "pkg", "version": "1.0"}]}), encoding="utf-8")
    status, failures, meta = check_sbom(path)
    assert status == "PASS"
    assert not failures
    assert meta["component_count"] == 1


def test_security_policy_passes_clean_bandit(tmp_path: Path) -> None:
    report_dir = tmp_path / "security"
    report_dir.mkdir()
    (report_dir / "bandit.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (report_dir / "pip_audit.json").write_text(json.dumps([]), encoding="utf-8")
    (report_dir / "detect_secrets.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (report_dir / "detect_secrets_audit.txt").write_text("No issues", encoding="utf-8")

    bandit_status, bandit_fail, _ = check_bandit(json.loads((report_dir / "bandit.json").read_text()), [])
    pip_status, pip_fail, _ = check_pip_audit(json.loads((report_dir / "pip_audit.json").read_text()), [])
    assert bandit_status == "PASS"
    assert pip_status == "PASS"
    assert not bandit_fail
    assert not pip_fail
