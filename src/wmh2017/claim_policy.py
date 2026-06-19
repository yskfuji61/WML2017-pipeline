"""Manager report claim policy helpers (v4)."""

from __future__ import annotations

import re

PROHIBITED_PATTERNS = [
    re.compile(r"\bREADY_FOR_RELEASE\b", re.I),
    re.compile(r"\bSOTA\b", re.I),
    re.compile(r"\bclinical(?:ly)? ready\b", re.I),
    re.compile(r"\bcustomer[- ]ready\b", re.I),
    re.compile(r"\bproduction[- ]ready\b", re.I),
    re.compile(r"臨床利用可能"),
    re.compile(r"顧客提示可能"),
    re.compile(r"本番利用可能"),
    re.compile(r"AI診断"),
    re.compile(r"official benchmark equivalent", re.I),
]

ALLOWED_MARKERS = (
    "not ready_for_release",
    "must never",
    "prohibited",
    "blocked",
    "local validation",
    "no_real_run_evidence",
    "missing",
)


def scan_report_text(text: str) -> list[str]:
    hits: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(marker in lower for marker in ALLOWED_MARKERS):
            continue
        for pattern in PROHIBITED_PATTERNS:
            if pattern.search(line):
                hits.append(f"line {line_no}: {line.strip()[:160]}")
    return hits
