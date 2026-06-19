#!/usr/bin/env python3
"""Detect absolute filesystem paths in run evidence and reports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ABS_PATH_PATTERN = re.compile(r"(?<![\w/.-])/(?:Users|home|tmp|var|opt|Volumes|private|mnt|data)/[^\s\"',}\]]+")


def scan_text(text: str, *, source: str) -> list[str]:
    hits: list[str] = []
    for match in ABS_PATH_PATTERN.finditer(text):
        hits.append(f"{source}: {match.group(0)}")
    return hits


def scan_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".pt", ".gz", ".nii"}:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return scan_text(text, source=path.as_posix())


def scan_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    hits: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            hits.extend(scan_file(path))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify no absolute path leakage in artifacts.")
    parser.add_argument("--run-dir", default="", help="Optional run directory to scan")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hits: list[str] = []
    reports_dir = repo_root / args.reports_dir
    hits.extend(scan_tree(reports_dir))

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        hits.extend(scan_tree(run_dir))

    if hits:
        raise SystemExit("absolute path leakage FAIL:\n" + "\n".join(hits[:50]))
    print("absolute path leakage gate PASS")


if __name__ == "__main__":
    main()
