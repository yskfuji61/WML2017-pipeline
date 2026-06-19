"""Overclaim scanner with negative-context allowance (v4 policy)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SCAN_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".csv", ".json", ".txt", ".rst", ".toml"}
SKIP_PATH_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "mlruns",
    "artifacts/runs",
    "node_modules",
}
SKIP_FILES = {
    "registry/claim_wording_policy.csv",
    "registry/claim_register_wmh2017.csv",
    "registry/claim_evidence_map.json",
    "tests/security/test_no_overclaim_wording.py",
    "tests/security/test_overclaim_guard.py",
    "tests/integration/test_minimal_pipeline_fixture.py",
    "docs/future_sota/EXP-001_winner_reproduction_plan.md",
    "docs/future_sota/sota_strategy_wmh2017.md",
    "src/wmh2017/security/overclaim.py",
    "src/wmh2017/registry/release_state.py",
    "src/wmh2017/claim_policy.py",
    "src/wmh2017/evidence.py",
}


class ClaimContext(str, Enum):
    PROHIBITED_POSITIVE = "prohibited_positive_claim"
    ALLOWED_NEGATIVE = "allowed_negative_boundary"
    ALLOWED_POLICY = "allowed_policy_name"
    ALLOWED_HISTORICAL = "allowed_historical_reference"
    REVIEW_REQUIRED = "review_required_ambiguous"


@dataclass(frozen=True)
class OverclaimHit:
    path: str
    line_no: int
    line: str
    pattern_id: str
    context: ClaimContext


PROHIBITED_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("ready_for_release_positive", re.compile(r"\bREADY_FOR_RELEASE\b", re.I)),
    (
        "clinical_ready_en",
        re.compile(r"\b(?:clinically validated|clinical(?:ly)? ready|clinical use is (?:approved|allowed))\b", re.I),
    ),
    ("customer_ready_en", re.compile(r"\bcustomer[- ]ready\b", re.I)),
    ("production_ready_en", re.compile(r"\bproduction[- ]ready\b", re.I)),
    ("sota_positive_en", re.compile(r"\b(?:this (?:is )?SOTA|SOTA achieved|state of the art)\b", re.I)),
    ("clinical_ready_ja", re.compile(r"臨床利用可能")),
    ("customer_ready_ja", re.compile(r"顧客提示可能")),
    ("production_ready_ja", re.compile(r"本番利用可能")),
    ("ai_diagnosis_ja", re.compile(r"AI診断(?:できます|可能)")),
    ("best_accuracy_ja", re.compile(r"最高精度")),
]

NEGATIVE_MARKERS = (
    "not ready_for_release",
    "not ready for release",
    "not `ready_for_release`",
    "must never be claimed",
    "must never be inferred",
    "remains blocked",
    "remain blocked",
    "forbidden",
    "prohibited",
    "blocked",
    "out of scope",
    "ready_for_release: false",
    "impossible",
    "not approved",
    "do not call",
    "do not claim",
    "do not overclaim",
    "non_release",
    "it is not",
    "preview-ready, customer-ready",
    "clinical, or production-ready",
    "or production-ready.",
    "without sota",
    "no sota claim",
    "sota overclaim",
    "sota wording",
)

POLICY_MARKERS = (
    "release_ladder",
    "release_state",
    "controlled_release_state",
    "claim_boundary",
    "claim register",
    "claim_wording_policy",
    "manager report",
    "prohibited_wording",
    "allowed_wording",
    "forbidden interpretation",
    "state mapping",
    "upper_bound",
    "preview_states",
    "require_all",
    "prohibited states",
    "prohibited claims",
    "state enum",
)


def _normalize_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def should_scan_file(path: Path, root: Path) -> bool:
    rel = _normalize_rel(path, root)
    if rel in SKIP_FILES:
        return False
    if path.suffix.lower() not in SCAN_EXTENSIONS:
        return False
    parts = set(rel.split("/"))
    if parts & SKIP_PATH_PARTS:
        return False
    return True


def classify_line(line: str, pattern_id: str) -> ClaimContext:
    lower = line.lower()
    stripped = line.strip()

    if '"ready_for_release": false' in lower or "'ready_for_release': false" in lower:
        return ClaimContext.ALLOWED_NEGATIVE

    if pattern_id == "ready_for_release_positive":
        if any(
            token in lower
            for token in (
                "must never",
                "not approved",
                "non_release",
                "requires formal",
                "requires additional",
                "it is not",
                "ready_for_release must",
                "ready_for_release remains",
            )
        ):
            return ClaimContext.ALLOWED_NEGATIVE
        if stripped.startswith(("-", "|", "`")):
            return ClaimContext.ALLOWED_POLICY
        if any(token in lower for token in ("preview_states", "require_all", "prohibited states")):
            return ClaimContext.ALLOWED_POLICY

    if pattern_id in {"customer_ready_en", "production_ready_en", "clinical_ready_en"}:
        if stripped.startswith("-"):
            return ClaimContext.ALLOWED_NEGATIVE
        if any(
            token in lower
            for token in (
                "forbidden",
                "prohibited",
                "blocked",
                "not ",
                "no ",
                "without ",
                "do not",
                "overclaim",
                "preview-ready, customer-ready",
                "clinical, or production-ready",
                "or production-ready.",
            )
        ):
            return ClaimContext.ALLOWED_NEGATIVE

    if pattern_id == "sota_positive_en":
        if any(
            token in lower
            for token in ("do not", "not ", "without ", "no sota", "overclaim", "forbidden", "prohibited")
        ):
            return ClaimContext.ALLOWED_NEGATIVE

    if any(marker in lower for marker in NEGATIVE_MARKERS):
        return ClaimContext.ALLOWED_NEGATIVE
    if any(marker in lower for marker in POLICY_MARKERS):
        return ClaimContext.ALLOWED_POLICY
    if "historical" in lower or "legacy" in lower:
        return ClaimContext.ALLOWED_HISTORICAL
    return ClaimContext.PROHIBITED_POSITIVE


def scan_text(text: str, *, source: str) -> list[OverclaimHit]:
    hits: list[OverclaimHit] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern_id, pattern in PROHIBITED_RULES:
            if not pattern.search(line):
                continue
            context = classify_line(line, pattern_id)
            if context == ClaimContext.PROHIBITED_POSITIVE:
                hits.append(
                    OverclaimHit(
                        path=source,
                        line_no=line_no,
                        line=line.strip(),
                        pattern_id=pattern_id,
                        context=context,
                    )
                )
    return hits


def scan_file(path: Path, root: Path) -> list[OverclaimHit]:
    if not should_scan_file(path, root):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    rel = _normalize_rel(path, root)
    return scan_text(text, source=rel)


def scan_tree(root: Path) -> list[OverclaimHit]:
    hits: list[OverclaimHit] = []
    for path in root.rglob("*"):
        if path.is_file():
            hits.extend(scan_file(path, root))
    return hits
