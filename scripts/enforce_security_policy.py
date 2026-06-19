#!/usr/bin/env python3
"""Fail-closed security policy enforcement for WMH2017 pipeline (v2 schema)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from wmh2017.security.scan_report import (
    ScanReportError,
    assert_scan_completed,
    load_json_report,
    validate_report_schema,
)

BANDIT_FAIL_SEVERITIES = {"HIGH", "MEDIUM"}
BANDIT_FAIL_CONFIDENCE = {"HIGH", "MEDIUM"}
PIP_AUDIT_FAIL_SEVERITIES = {"CRITICAL", "HIGH"}


def _load_exceptions(register_path: Path) -> list[dict[str, str]]:
    if not register_path.exists():
        return []
    with register_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exception_covers(exceptions: list[dict[str, str]], *, tool: str, finding_id: str, severity: str) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    for row in exceptions:
        if row.get("status", "").upper() != "APPROVED":
            continue
        if row.get("tool", "") != tool:
            continue
        registered_id = row.get("finding_id", "")
        if registered_id and registered_id != finding_id:
            if ":*:" in registered_id:
                prefix, _, package = registered_id.partition(":*:")
                if not (finding_id.startswith(prefix + ":") and finding_id.endswith(":" + package)):
                    continue
            else:
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


def check_bandit(report: dict, exceptions: list[dict[str, str]]) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    high = medium = 0
    for item in report.get("results", []):
        severity = str(item.get("issue_severity", "")).upper()
        confidence = str(item.get("issue_confidence", "")).upper()
        if severity == "HIGH":
            high += 1
        if severity == "MEDIUM":
            medium += 1
        if severity not in BANDIT_FAIL_SEVERITIES or confidence not in BANDIT_FAIL_CONFIDENCE:
            continue
        test_id = str(item.get("test_id", "unknown"))
        finding_id = f"bandit:{test_id}:{item.get('line_number', '')}"
        if _exception_covers(exceptions, tool="bandit", finding_id=finding_id, severity=severity):
            continue
        failures.append(f"bandit {severity}/{confidence}: {test_id}")
    status = "PASS" if not failures else "FAIL"
    return status, failures, {"high_findings": high, "medium_high_confidence_findings": medium}


def check_pip_audit(report: dict | list, exceptions: list[dict[str, str]]) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    critical = high = 0
    deps = report if isinstance(report, list) else report.get("dependencies", [])
    for dep in deps:
        for vuln in dep.get("vulns", []):
            sev = str(vuln.get("severity", "HIGH")).upper()
            if sev == "CRITICAL":
                critical += 1
            if sev == "HIGH":
                high += 1
            if sev not in PIP_AUDIT_FAIL_SEVERITIES:
                continue
            alias = vuln.get("id") or (vuln.get("aliases") or ["unknown"])[0]
            finding_id = f"pip-audit:{alias}:{dep.get('name', '')}"
            if _exception_covers(exceptions, tool="pip-audit", finding_id=finding_id, severity=sev):
                continue
            failures.append(f"pip-audit {sev}: {dep.get('name')} {alias}")
    status = "PASS" if not failures else "FAIL"
    return (
        status,
        failures,
        {
            "critical_vulnerabilities": critical,
            "high_vulnerabilities_without_exception": len(failures),
        },
    )


def check_detect_secrets(
    report_dir: Path, audit_txt: Path, exceptions: list[dict[str, str]]
) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    unaudited = 0
    detect_json = report_dir / "detect_secrets.json"
    data = load_json_report(detect_json)
    validate_report_schema(data, "detect-secrets")
    if isinstance(data, dict):
        for item in data.get("results", []):
            finding_id = f"secret:{item.get('type')}:{item.get('filename')}:{item.get('line_number')}"
            if _exception_covers(exceptions, tool="detect-secrets", finding_id=finding_id, severity="HIGH"):
                continue
            failures.append(f"detect-secrets: {item.get('type')} in {item.get('filename')}")
    if not audit_txt.exists():
        failures.append(f"missing detect-secrets audit report: {audit_txt}")
    else:
        text = audit_txt.read_text(encoding="utf-8")
        if "Unaudited secrets" in text or "Potential secrets" in text:
            unaudited += 1
            if not _exception_covers(
                exceptions, tool="detect-secrets", finding_id="baseline:unaudited", severity="HIGH"
            ):
                failures.append("detect-secrets audit: unaudited baseline entries remain")
    status = "PASS" if not failures else "FAIL"
    return status, failures, {"unaudited_findings": unaudited}


def check_sbom(path: Path) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    try:
        data = load_json_report(path)
        validate_report_schema(data, "sbom")
    except ScanReportError as exc:
        failures.append(str(exc))
        return "FAIL", failures, {"path": str(path), "component_count": 0}
    components = data.get("components", []) if isinstance(data, dict) else []
    if not components:
        failures.append("SBOM empty")
    status = "PASS" if not failures else "FAIL"
    return status, failures, {"path": str(path), "component_count": len(components)}


def enforce_security_policy(
    report_dir: Path,
    *,
    repo_root: Path,
    exception_register: Path,
    sbom_path: Path,
) -> tuple[dict, list[str]]:
    exceptions = _load_exceptions(exception_register)
    all_failures: list[str] = []

    for tool in ("detect-secrets", "bandit", "pip-audit"):
        try:
            assert_scan_completed(report_dir, tool)
        except ScanReportError as exc:
            all_failures.append(str(exc))

    try:
        ds_status, ds_fail, ds_meta = check_detect_secrets(
            report_dir,
            report_dir / "detect_secrets_audit.txt",
            exceptions,
        )
    except ScanReportError as exc:
        ds_status, ds_fail, ds_meta = "FAIL", [str(exc)], {}
    all_failures.extend(ds_fail)

    try:
        bandit_report_raw = load_json_report(report_dir / "bandit.json")
        validate_report_schema(bandit_report_raw, "bandit")
        assert isinstance(bandit_report_raw, dict)
        bandit_status, bandit_fail, bandit_meta = check_bandit(bandit_report_raw, exceptions)
    except ScanReportError as exc:
        bandit_status, bandit_fail, bandit_meta = "FAIL", [str(exc)], {}
    all_failures.extend(bandit_fail)

    try:
        pip_report = load_json_report(report_dir / "pip_audit.json")
        validate_report_schema(pip_report, "pip-audit")
        pip_status, pip_fail, pip_meta = check_pip_audit(pip_report, exceptions)
    except ScanReportError as exc:
        pip_status, pip_fail, pip_meta = "FAIL", [str(exc)], {}
    all_failures.extend(pip_fail)

    sbom_status, sbom_fail, sbom_meta = check_sbom(sbom_path)
    all_failures.extend(sbom_fail)

    result = {
        "status": "PASS" if not all_failures else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "detect_secrets": {"status": ds_status, **ds_meta},
        "bandit": {"status": bandit_status, **bandit_meta},
        "pip_audit": {"status": pip_status, **pip_meta},
        "sbom": {"status": sbom_status, **sbom_meta},
        "exceptions": [],
        "failures": all_failures,
    }
    return result, all_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce fail-closed security policy.")
    parser.add_argument("report_dir", help="Directory containing security scan JSON outputs")
    parser.add_argument("--exception-register", default="registry/security_exception_register.csv")
    parser.add_argument("--sbom", default="reports/security/sbom.cdx.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = repo_root / report_dir

    result, all_failures = enforce_security_policy(
        report_dir,
        repo_root=repo_root,
        exception_register=repo_root / args.exception_register,
        sbom_path=repo_root / args.sbom,
    )

    out_path = Path(args.out) if args.out else report_dir / "security_policy_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_failures:
        raise SystemExit(f"security policy FAIL ({len(all_failures)} issues):\n" + "\n".join(all_failures))
    print(f"security policy PASS -> {out_path}")


if __name__ == "__main__":
    main()
