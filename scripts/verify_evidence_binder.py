#!/usr/bin/env python3
"""Verify evidence binder closure for target release state."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def _load_findings(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evidence binder closure.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--binder", default="registry/evidence_binder_wmh2017.yaml")
    parser.add_argument("--finding-register", default="registry/finding_register_wmh2017.csv")
    parser.add_argument("--review-register", default="registry/review_approval_register_wmh2017.csv")
    parser.add_argument("--package-manifest", default="reports/full_package_manifest.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    binder_path = repo_root / args.binder
    if not binder_path.exists():
        raise SystemExit(f"missing evidence binder: {binder_path}")

    binder = yaml.safe_load(binder_path.read_text(encoding="utf-8")) or {}
    failures: list[str] = []

    pkg_manifest = repo_root / args.package_manifest
    if binder.get("required_sections", {}).get("package_identity", {}).get("required"):
        if not pkg_manifest.exists():
            failures.append(f"missing package manifest: {pkg_manifest}")

    source_artifact = binder.get("required_sections", {}).get("source_register", {}).get("artifact", "")
    if source_artifact and not (repo_root / source_artifact).exists():
        failures.append(f"missing source register artifact: {source_artifact}")

    if args.run_id:
        run_dir = repo_root / "artifacts" / "runs" / args.run_id
        required_artifacts = binder.get("required_sections", {}).get("real_run", {}).get("required_artifacts", [])
        for pattern in required_artifacts:
            rel = pattern.replace("{run_id}", args.run_id)
            path = repo_root / rel
            if not path.exists():
                failures.append(f"missing run artifact: {rel}")

        manifest = run_dir / "artifact_manifest.json"
        if not manifest.exists():
            failures.append(f"missing artifact manifest for run: {manifest}")

    security_section = binder.get("required_sections", {}).get("security", {})
    if security_section.get("fail_closed"):
        for rel in security_section.get("required_artifacts", []):
            if not (repo_root / rel).exists():
                failures.append(f"missing security artifact: {rel}")

    findings = _load_findings(repo_root / args.finding_register)
    open_sev0_sev1 = [
        f["finding_id"]
        for f in findings
        if f.get("severity", "") in {"Sev0", "Sev1"} and f.get("status", "").upper().startswith("OPEN")
    ]
    if open_sev0_sev1 and args.target_state == "READY_FOR_PREVIEW":
        failures.append(f"unresolved Sev0/Sev1 findings: {open_sev0_sev1}")

    if failures:
        raise SystemExit("evidence binder gate FAIL:\n" + "\n".join(failures))
    print(f"evidence binder gate PASS for target {args.target_state}")


if __name__ == "__main__":
    main()
