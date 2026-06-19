from scripts.enforce_security_policy import check_bandit


def test_enforce_security_policy_fails_on_high_bandit_finding():
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
