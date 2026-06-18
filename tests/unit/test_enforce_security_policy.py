import json
from pathlib import Path

from scripts.enforce_security_policy import check_bandit, check_pip_audit


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
    failures = check_bandit(report, [])
    assert failures


def test_pip_audit_critical_fails() -> None:
    report = [{"name": "badpkg", "vulns": [{"id": "CVE-TEST-001", "severity": "CRITICAL"}]}]
    failures = check_pip_audit(report, [])
    assert failures


def test_security_policy_passes_clean_bandit(tmp_path: Path) -> None:
    report_dir = tmp_path / "security"
    report_dir.mkdir()
    (report_dir / "bandit.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (report_dir / "pip_audit.json").write_text(json.dumps([]), encoding="utf-8")
    (report_dir / "detect_secrets.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (report_dir / "detect_secrets_audit.txt").write_text("No issues", encoding="utf-8")

    assert not check_bandit(json.loads((report_dir / "bandit.json").read_text()), [])
    assert not check_pip_audit(json.loads((report_dir / "pip_audit.json").read_text()), [])
