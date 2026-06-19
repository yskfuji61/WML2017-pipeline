#!/usr/bin/env python3
"""Fail if git-tracked files include raw medical imaging data."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".nii",
    ".nii.gz",
    ".dcm",
    ".mhd",
    ".raw",
    ".img",
    ".hdr",
}

FORBIDDEN_PREFIXES = ("Datasets/", "data/raw/")


def git_tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files"],
        cwd=str(repo_root),
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_forbidden(rel: str) -> str | None:
    lower = rel.lower()
    for suffix in FORBIDDEN_SUFFIXES:
        if lower.endswith(suffix):
            return f"forbidden suffix {suffix}"
    for prefix in FORBIDDEN_PREFIXES:
        if rel.startswith(prefix):
            return f"forbidden prefix {prefix}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify no raw medical data is committed.")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    offenders: list[str] = []
    for rel in git_tracked_files(repo_root):
        reason = is_forbidden(rel)
        if reason:
            offenders.append(f"{rel} ({reason})")

    if offenders:
        raise SystemExit(
            "raw data gate FAIL: tracked files must not include medical imaging data:\n" + "\n".join(offenders)
        )
    print("raw data gate PASS")


if __name__ == "__main__":
    main()
