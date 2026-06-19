#!/usr/bin/env python3
"""Validate run artifact JSON files against registry schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmh2017.registry.schema_validation import validate_required_keys


def validate_run_artifacts(repo_root: Path, run_id: str) -> list[str]:
    failures: list[str] = []
    run_dir = repo_root / "artifacts/runs" / run_id
    checks = [
        (run_dir / "artifact_manifest.json", "artifact_manifest.schema.json"),
        (run_dir / "lineage/lineage_graph.json", "lineage_graph.schema.json"),
        (run_dir / "run_evidence.json", "run_evidence.schema.json"),
    ]
    for path, schema_name in checks:
        if not path.exists():
            failures.append(f"missing artifact for schema validation: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            failures.append(f"expected JSON object: {path}")
            continue
        failures.extend(f"{path.name}: {msg}" for msg in validate_required_keys(payload, schema_name))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate run artifact schemas.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    failures = validate_run_artifacts(repo_root, args.run_id)
    if failures:
        raise SystemExit("artifact schema validation FAIL:\n" + "\n".join(failures))
    print("artifact schema validation PASS")


if __name__ == "__main__":
    main()
