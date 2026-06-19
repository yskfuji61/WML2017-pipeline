#!/usr/bin/env python3
"""Verify official evaluator pin against source record and tree hash."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def verify_official_evaluator_source(repo_root: Path, *, structure_only: bool = False) -> list[str]:
    failures: list[str] = []
    pin_path = repo_root / "registry/official_evaluator_pin.yaml"
    source_record = repo_root / "third_party/official_wmh_evaluator/SOURCE_RECORD.md"
    exact_commit = repo_root / "third_party/official_wmh_evaluator/exact_commit_or_archive.txt"
    evaluator_sha = repo_root / "third_party/official_wmh_evaluator/evaluator.sha256"

    if not pin_path.exists():
        return [f"missing pin file: {pin_path}"]

    pin = yaml.safe_load(pin_path.read_text(encoding="utf-8")) or {}
    commit = str(pin.get("commit", ""))
    expected_tree = str(pin.get("expected_tree_sha256", ""))
    license_status = str(pin.get("license_review_status", ""))

    if commit == "PENDING" or expected_tree == "PENDING":
        failures.append("evaluator pin not fixed (commit or expected_tree_sha256 is PENDING)")
        if structure_only:
            return failures
    elif not COMMIT_RE.match(commit):
        failures.append(f"invalid pinned commit: {commit}")
    elif not SHA256_RE.match(expected_tree):
        failures.append(f"invalid expected_tree_sha256: {expected_tree}")

    if license_status.upper() != "APPROVED":
        failures.append(f"license_review_status is {license_status}, not APPROVED")

    if source_record.exists():
        text = source_record.read_text(encoding="utf-8")
        if "NOT_FETCHED" in text or "PENDING" in text:
            failures.append("SOURCE_RECORD.md indicates evaluator not fetched")
        if commit != "PENDING" and f"Commit: {commit}" not in text:
            failures.append("SOURCE_RECORD commit does not match pin")
    else:
        failures.append(f"missing source record: {source_record}")

    if commit != "PENDING":
        if exact_commit.exists():
            recorded = exact_commit.read_text(encoding="utf-8").strip()
            if recorded != commit:
                failures.append("exact_commit_or_archive.txt does not match pin commit")
        else:
            failures.append(f"missing exact commit file: {exact_commit}")

        if evaluator_sha.exists():
            recorded_hash = evaluator_sha.read_text(encoding="utf-8").strip()
            if expected_tree != "PENDING" and recorded_hash != expected_tree:
                failures.append("evaluator.sha256 does not match pin expected_tree_sha256")
        elif not structure_only:
            failures.append(f"missing evaluator hash file: {evaluator_sha}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify official evaluator supply chain.")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    failures = verify_official_evaluator_source(repo_root, structure_only=args.structure_only)
    if failures:
        raise SystemExit("official evaluator source FAIL:\n" + "\n".join(failures))
    print("official evaluator source PASS")


if __name__ == "__main__":
    main()
