#!/usr/bin/env python3
"""Fail-closed security policy enforcement for WMH2017 pipeline (v2 schema)."""
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


def _exception_covers(exceptions: list[dict[str, str]], *, tool: str, finding_id: str, severity: str) -> bool:
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
    return status, failures, {
        "critical_vulnerabilities": critical,
        "high_vulnerabilities_without_exception": len(failures),
    }


def check_detect_secrets(report_dir: Path, audit_txt: Path, exceptions: list[dict[str, str]]) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    unaudited = 0
    detect_json = report_dir / "detect_secrets.json"
    if detect_json.exists():
        data = _load_json(detect_json)
        if isinstance(data, dict):
            for item in data.get("results", []):
                finding_id = f"secret:{item.get('type')}:{item.get('filename')}:{item.get('line_number')}"
                if _exception_covers(exceptions, tool="detect-secrets", finding_id=finding_id, severity="HIGH"):
                    continue
                failures.append(f"detect-secrets: {item.get('type')} in {item.get('filename')}")
    if audit_txt.exists():
        text = audit_txt.read_text(encoding="utf-8")
        if "Unaudited secrets" in text or "Potential secrets" in text:
            unaudited += 1
            if not _exception_covers(exceptions, tool="detect-secrets", finding_id="baseline:unaudited", severity="HIGH"):
                failures.append("detect-secrets audit: unaudited baseline entries remain")
    status = "PASS" if not failures else "FAIL"
    return status, failures, {"unaudited_findings": unaudited}


def check_sbom(path: Path) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    if not path.exists():
        failures.append(f"SBOM missing: {path}")
        return "FAIL", failures, {"path": str(path), "component_count": 0}
    data = _load_json(path)
    components = data.get("components", []) if isinstance(data, dict) else []
    if not components:
        failures.append("SBOM empty")
    status = "PASS" if not failures else "FAIL"
    return status, failures, {"path": str(path), "component_count": len(components)}


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
    exceptions = _load_exceptions(repo_root / args.exception_register)

    all_failures: list[str] = []
    ds_status, ds_fail, ds_meta = check_detect_secrets(report_dir, report_dir / "detect_secrets_audit.txt", exceptions)
    all_failures.extend(ds_fail)
    bandit_status, bandit_fail, bandit_meta = check_bandit(_load_json(report_dir / "bandit.json"), exceptions)
    all_failures.extend(bandit_fail)
    pip_status, pip_fail, pip_meta = check_pip_audit(_load_json(report_dir / "pip_audit.json"), exceptions)
    all_failures.extend(pip_fail)
    sbom_path = repo_root / args.sbom
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

    out_path = Path(args.out) if args.out else report_dir / "security_policy_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_failures:
        raise SystemExit(f"security policy FAIL ({len(all_failures)} issues):\n" + "\n".join(all_failures))
    print(f"security policy PASS -> {out_path}")


if __name__ == "__main__":
    main()
