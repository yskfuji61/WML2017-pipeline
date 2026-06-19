#!/usr/bin/env python3
"""Detect credential-like strings in git-tracked text files."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".csv", ".txt", ".toml", ".sh", ".env.example"}
SKIP_PREFIXES = (
    ".secrets.baseline",
    "reports/security/detect_secrets",
    "tests/",
)
SKIP_FILES = {
    "scripts/verify_no_env_secrets.py",
    "registry/claim_wording_policy.csv",
}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("generic_api_key", re.compile(r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./]{16,}")),
    ("password_assignment", re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{20,}")),
]


def git_tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files"],
        cwd=str(repo_root),
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def should_scan(rel: str) -> bool:
    if rel in SKIP_FILES:
        return False
    if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
        return False
    return Path(rel).suffix.lower() in TEXT_SUFFIXES or rel.endswith(".env.example")


def scan_file(repo_root: Path, rel: str) -> list[str]:
    path = repo_root / rel
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    hits: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "PENDING_CONFIRMATION" in line or "REDACTED" in line or "example" in line.lower():
            continue
        for pattern_id, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append(f"{rel}:{line_no} [{pattern_id}]")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify no env secrets in tracked files.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    offenders: list[str] = []
    for rel in git_tracked_files(repo_root):
        if should_scan(rel):
            offenders.extend(scan_file(repo_root, rel))

    if offenders:
        raise SystemExit("env secret gate FAIL:\n" + "\n".join(offenders[:50]))
    print("env secret gate PASS")


if __name__ == "__main__":
    main()
