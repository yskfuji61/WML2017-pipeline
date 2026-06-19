#!/usr/bin/env python3
"""Fail if remote MLflow/cloud tracking URIs are enabled by default."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".json", ".toml", ".env.example"}
SKIP_PARTS = {".git", "venv", ".venv", "__pycache__", "artifacts/runs", "mlruns"}
SKIP_FILES = {
    "scripts/verify_no_unapproved_remote_uri.py",
}

REMOTE_URI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mlflow_remote_http", re.compile(r"MLFLOW_TRACKING_URI\s*[:=]\s*['\"]https?://", re.I)),
    ("mlflow_databricks", re.compile(r"databricks", re.I)),
    ("mlflow_remote_azure", re.compile(r"azureml://", re.I)),
    ("mlflow_remote_arn", re.compile(r"arn:aws:sagemaker:", re.I)),
    ("default_remote_tracking", re.compile(r"tracking_uri\s*:\s*['\"]https?://", re.I)),
]

ALLOWED_MARKERS = (
    "file://",
    "file:./",
    "local only",
    "local-only",
    "refused",
    "prohibited",
    "must not",
    "not approved",
    "example only",
    "PENDING_CONFIRMATION",
)


def should_scan(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in SKIP_FILES:
        return False
    if any(part in rel for part in SKIP_PARTS):
        return False
    return path.suffix.lower() in SCAN_SUFFIXES


def scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    hits: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in ALLOWED_MARKERS):
            continue
        for pattern_id, pattern in REMOTE_URI_PATTERNS:
            if pattern.search(line):
                hits.append(f"{path.as_posix()}:{line_no} [{pattern_id}] {line.strip()[:120]}")
    return hits


def scan_tree(root: Path) -> list[str]:
    hits: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and should_scan(path, root):
            hits.extend(scan_file(path))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify no unapproved remote tracking URIs.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    hits = scan_tree(root)
    if hits:
        raise SystemExit("remote URI gate FAIL:\n" + "\n".join(hits[:50]))
    print("remote URI gate PASS")


if __name__ == "__main__":
    main()
