#!/usr/bin/env python3
"""Fail-closed security policy enforcement for WMH2017 pipeline."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

BANDIT_FAIL_SEVERITIES = {"HIGH", "MEDIUM"}
BANDIT_FAIL_CONFIDENCE = {"HIGH", "MEDIUM"}
PIP_AUDIT_FAIL_SEVERITIES = {"CRITICAL", "HIGH"}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _load_exceptions(register_path: Path) -> list[dict[str, str]]:
    if not register_path.exists():
        return []
    with register_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exception_covers(
    exceptions: list[dict[str, str]],
    *,
    tool: str,
    finding_id: str,
    severity: str,
) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    for row in exceptions:
        if row.get("status", "").upper() != "APPROVED":
            continue
        if row.get("tool", "") != tool:
            continue
        if row.get("finding_id", "") and row.get("finding_id") != finding_id:
            continue
        if row.get("severity", "").upper() and row.get("severity", "").upper() != severity.upper():
            continue
        expiry = row.get("expiry_date", "")
        if expiry and expiry < today:
            continue
        required = ("owner", "rationale", "compensating_control", "reviewer")
        if all(row.get(k, "").strip() for k in required):
            return True
    return False


def check_bandit(report: dict, exceptions: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    for item in report.get("results", []):
        severity = str(item.get("issue_severity", "")).upper()
        confidence = str(item.get("issue_confidence", "")).upper()
        if severity not in BANDIT_FAIL_SEVERITIES or confidence not in BANDIT_FAIL_CONFIDENCE:
            continue
        test_id = str(item.get("test_id", "unknown"))
        finding_id = f"bandit:{test_id}:{item.get('line_number', '')}"
        if _exception_covers(exceptions, tool="bandit", finding_id=finding_id, severity=severity):
            continue
        failures.append(f"bandit {severity}/{confidence}: {test_id} at {item.get('filename')}:{item.get('line_number')}")
    return failures


def check_pip_audit(report: dict | list, exceptions: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    deps = report if isinstance(report, list) else report.get("dependencies", [])
    for dep in deps:
        name = dep.get("name", "unknown")
        for vuln in dep.get("vulns", []):
            alias = vuln.get("id") or (vuln.get("aliases") or ["unknown"])[0]
            severity_raw = str(vuln.get("severity", "HIGH")).upper()
            if severity_raw not in PIP_AUDIT_FAIL_SEVERITIES:
                continue
            finding_id = f"pip-audit:{alias}:{name}"
            if _exception_covers(exceptions, tool="pip-audit", finding_id=finding_id, severity=severity_raw):
                continue
            failures.append(f"pip-audit {severity_raw}: {name} {alias}")
    return failures


def check_detect_secrets(report_dir: Path, audit_txt: Path, exceptions: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    detect_json = report_dir / "detect_secrets.json"
    if detect_json.exists():
        data = _load_json(detect_json)
        if isinstance(data, dict):
            for item in data.get("results", []):
                finding_id = f"secret:{item.get('type')}:{item.get('filename')}:{item.get('line_number')}"
                if _exception_covers(exceptions, tool="detect-secrets", finding_id=finding_id, severity="HIGH"):
                    continue
                failures.append(
                    f"detect-secrets: {item.get('type')} in {item.get('filename')}:{item.get('line_number')}"
                )
    if audit_txt.exists():
        text = audit_txt.read_text(encoding="utf-8")
        if "Unaudited secrets" in text or "Potential secrets" in text:
            if not _exception_covers(exceptions, tool="detect-secrets", finding_id="baseline:unaudited", severity="HIGH"):
                failures.append("detect-secrets audit: unaudited baseline entries remain")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce fail-closed security policy.")
    parser.add_argument("report_dir", help="Directory containing security scan JSON outputs")
    parser.add_argument(
        "--exception-register",
        default="registry/security_exception_register.csv",
        help="Approved exception register path",
    )
    parser.add_argument("--out", default="", help="Optional security_policy_result.json path")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    repo_root = Path(__file__).resolve().parents[1]
    exception_path = Path(args.exception_register)
    if not exception_path.is_absolute():
        exception_path = repo_root / exception_path

    exceptions = _load_exceptions(exception_path)
    failures: list[str] = []
    failures.extend(check_bandit(_load_json(report_dir / "bandit.json"), exceptions))
    failures.extend(check_pip_audit(_load_json(report_dir / "pip_audit.json"), exceptions))
    failures.extend(check_detect_secrets(report_dir, report_dir / "detect_secrets_audit.txt", exceptions))

    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "policy": {
            "bandit_severity": sorted(BANDIT_FAIL_SEVERITIES),
            "bandit_confidence": sorted(BANDIT_FAIL_CONFIDENCE),
            "pip_audit_severity": sorted(PIP_AUDIT_FAIL_SEVERITIES),
        },
    }

    out_path = Path(args.out) if args.out else report_dir / "security_policy_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if failures:
        raise SystemExit(f"security policy FAIL ({len(failures)} issues):\n" + "\n".join(failures))
    print(f"security policy PASS -> {out_path}")


if __name__ == "__main__":
    main()
