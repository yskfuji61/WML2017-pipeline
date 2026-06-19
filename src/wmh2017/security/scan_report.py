"""Security scan report loading and completion validation."""
from __future__ import annotations

import json
from pathlib import Path

COMPLETION_MARKERS = {
    "bandit": ".bandit.completed",
    "pip-audit": ".pip_audit.completed",
    "detect-secrets": ".detect_secrets.completed",
}


class ScanReportError(Exception):
    """Raised when a security scan report is missing, invalid, or incomplete."""


def load_json_report(path: Path) -> dict | list:
    if not path.exists():
        raise ScanReportError(f"missing report file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ScanReportError(f"empty report file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScanReportError(f"invalid JSON in report {path}: {exc}") from exc


def validate_report_schema(report: dict | list, tool: str) -> None:
    if tool == "bandit":
        if not isinstance(report, dict):
            raise ScanReportError("bandit report must be a JSON object")
        if "results" not in report:
            raise ScanReportError("bandit report missing results key")
        return
    if tool == "pip-audit":
        if isinstance(report, list):
            return
        if isinstance(report, dict) and "dependencies" in report:
            return
        raise ScanReportError("pip-audit report must be a JSON array or object with dependencies")
    if tool == "detect-secrets":
        if not isinstance(report, dict):
            raise ScanReportError("detect-secrets report must be a JSON object")
        return
    if tool == "sbom":
        if not isinstance(report, dict):
            raise ScanReportError("SBOM report must be a JSON object")
        if "components" not in report:
            raise ScanReportError("SBOM report missing components key")
        return
    raise ScanReportError(f"unknown tool for schema validation: {tool}")


def assert_scan_completed(report_dir: Path, tool: str) -> None:
    marker_name = COMPLETION_MARKERS.get(tool)
    if marker_name is None:
        raise ScanReportError(f"unknown tool for completion marker: {tool}")
    marker = report_dir / marker_name
    if not marker.exists():
        raise ScanReportError(f"scan completion marker missing for {tool}: {marker}")
