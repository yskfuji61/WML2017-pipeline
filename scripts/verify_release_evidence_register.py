#!/usr/bin/env python3
"""Verify release evidence register CSV structure and optional artifact hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

REQUIRED_COLUMNS = [
    "evidence_id",
    "run_id",
    "commit_sha",
    "ci_run_url",
    "artifact_path",
    "artifact_sha256",
    "evidence_type",
    "reviewer",
    "decision",
    "created_at_utc",
]
VALID_DECISIONS = {"pass", "fail", "blocked"}
VALID_EVIDENCE_TYPES = {"test", "e2e", "parity", "review", "security"}
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_DATA_PATTERNS = ("/MICCAI2017_WMH/", "/Datasets/", "wmh.nii.gz")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(register_path: Path) -> list[dict[str, str]]:
    with register_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise SystemExit(
                f"release evidence register columns mismatch: expected {REQUIRED_COLUMNS}, got {reader.fieldnames}"
            )
        return [dict(row) for row in reader]


def verify_register(
    repo_root: Path,
    *,
    run_id: str = "",
    structure_only: bool = False,
) -> list[str]:
    register_path = repo_root / "registry/release_evidence_register_wmh2017.csv"
    if not register_path.exists():
        return [f"missing register: {register_path}"]

    failures: list[str] = []
    rows = _load_rows(register_path)
    if not rows:
        failures.append("release evidence register is empty")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        evidence_id = row["evidence_id"]
        if evidence_id in seen_ids:
            failures.append(f"line {index}: duplicate evidence_id {evidence_id}")
        seen_ids.add(evidence_id)

        if row["decision"] not in VALID_DECISIONS:
            failures.append(f"line {index}: invalid decision {row['decision']}")
        if row["evidence_type"] not in VALID_EVIDENCE_TYPES:
            failures.append(f"line {index}: invalid evidence_type {row['evidence_type']}")

        commit_sha = row["commit_sha"]
        if commit_sha != "PENDING" and not COMMIT_SHA_RE.match(commit_sha):
            failures.append(f"line {index}: invalid commit_sha {commit_sha}")

        artifact_sha256 = row["artifact_sha256"]
        if artifact_sha256 != "PENDING" and not SHA256_RE.match(artifact_sha256):
            failures.append(f"line {index}: invalid artifact_sha256 {artifact_sha256}")

        artifact_path = row["artifact_path"]
        if any(token in artifact_path for token in RAW_DATA_PATTERNS):
            failures.append(f"line {index}: artifact_path must not reference raw dataset paths: {artifact_path}")

        if run_id and row["run_id"] != run_id:
            failures.append(f"line {index}: run_id mismatch expected {run_id}, got {row['run_id']}")

        if row["evidence_type"] in {"review", "parity"} and not row["reviewer"].strip():
            if row["decision"] == "pass":
                failures.append(f"line {index}: reviewer required for pass on {row['evidence_type']}")

        path = repo_root / artifact_path
        if structure_only or not path.exists():
            continue
        if artifact_sha256 == "PENDING":
            continue
        actual = _sha256_file(path)
        if actual != artifact_sha256:
            failures.append(f"line {index}: sha256 mismatch for {artifact_path}")

    schema_path = repo_root / "src/wmh2017/registry/schemas/release_evidence.schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required_props = set(schema.get("required", []))
        if not required_props.issubset(set(REQUIRED_COLUMNS)):
            failures.append("schema required fields not covered by CSV columns")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify release evidence register.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    failures = verify_register(
        repo_root,
        run_id=args.run_id,
        structure_only=args.structure_only,
    )
    if failures:
        raise SystemExit("release evidence register FAIL:\n" + "\n".join(failures))
    print("release evidence register PASS")


if __name__ == "__main__":
    main()
