#!/usr/bin/env python3
"""Sync manifest hash references after make manifest (single pass, no re-manifest)."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_csv_field(path: Path, row_prefix: str, field: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    idx = header.index(field)
    out = [lines[0]]
    for line in lines[1:]:
        if line.startswith(row_prefix + ","):
            parts = line.split(",")
            parts[idx] = value
            line = ",".join(parts)
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _set_yaml_field(path: Path, field: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(rf"^{field}: .*$", f"{field}: sha256:{value}", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync manifest hash references.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    manifest = repo / "reports/full_package_manifest.json"
    finding = repo / "registry/finding_register_wmh2017.csv"
    review = repo / "registry/review_approval_register_wmh2017.csv"
    binder = repo / "registry/evidence_binder_wmh2017.yaml"
    decision = repo / "docs/release_decisions" / f"release_decision_{args.run_id}.yaml"

    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    if not decision.exists():
        raise SystemExit(f"missing release decision: {decision}")

    manifest_hash = _sha256(manifest)
    binder_hash = _sha256(binder)

    _set_csv_field(review, "REV-WMH-004", "artifact_hash", manifest_hash)
    _set_csv_field(finding, "FIND-WMH-001", "closure_hash", manifest_hash)
    _set_yaml_field(decision, "package_manifest_sha256", manifest_hash)
    _set_yaml_field(decision, "evidence_binder_sha256", binder_hash)

    review_hash = _sha256(review)
    _set_csv_field(finding, "FIND-WMH-004", "closure_hash", review_hash)
    _set_csv_field(finding, "FIND-WMH-009", "closure_hash", binder_hash)

    print(f"synced manifest hash: {manifest_hash}")
    print(f"synced review register snapshot: {review_hash}")
    print(f"synced evidence binder hash: {binder_hash}")


if __name__ == "__main__":
    main()
