#!/usr/bin/env python3
"""Verify package identity consistency across registry sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml


def _read_pyproject_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else ""


def _read_readme_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"package_version\s*\|\s*`([^`]+)`", text)
    return m.group(1) if m else ""


def _read_release_decision_fields(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for key in ("package_version", "package_id", "package_manifest_sha256"):
        m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify package identity consistency.")
    parser.add_argument("--identity", default="registry/package_identity_wmh2017.yaml")
    parser.add_argument("--check-git-tag", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    identity_path = repo_root / args.identity
    identity = yaml.safe_load(identity_path.read_text(encoding="utf-8")) or {}

    expected_id = str(identity.get("package_id", ""))
    expected_version = str(identity.get("package_version", ""))
    expected_target = str(identity.get("target_state", ""))
    failures: list[str] = []

    readme_v = _read_readme_version(repo_root / "README.md")
    if readme_v != expected_version:
        failures.append(f"README.md package_version={readme_v} != {expected_version}")

    pyproject_v = _read_pyproject_version(repo_root / "pyproject.toml")
    if pyproject_v != expected_version:
        failures.append(f"pyproject.toml version={pyproject_v} != {expected_version}")

    binder_path = repo_root / str(identity.get("identity_sources", {}).get("evidence_binder", ""))
    if binder_path.exists():
        binder = yaml.safe_load(binder_path.read_text(encoding="utf-8")) or {}
        if str(binder.get("package_id", "")) != expected_id:
            failures.append("evidence_binder package_id mismatch")
        if str(binder.get("package_version", "")) != expected_version:
            failures.append("evidence_binder package_version mismatch")
        if str(binder.get("target_state", "")) != expected_target:
            failures.append("evidence_binder target_state mismatch")

    release_path = repo_root / str(identity.get("identity_sources", {}).get("release_decision", ""))
    if release_path.exists():
        fields = _read_release_decision_fields(release_path)
        if fields.get("package_version") and fields["package_version"] != expected_version:
            failures.append(f"release_decision_record package_version={fields.get('package_version')}")

    manifest_path = repo_root / str(identity.get("identity_sources", {}).get("full_manifest", ""))
    manifest_sha256 = ""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("package_id", "")) != expected_id:
            failures.append("full_package_manifest package_id mismatch")
        if str(manifest.get("package_version", "")) != expected_version:
            failures.append(f"full_package_manifest package_version={manifest.get('package_version')}")
        h = hashlib.sha256()
        with manifest_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        manifest_sha256 = h.hexdigest()

    if release_path.exists() and manifest_sha256:
        fields = _read_release_decision_fields(release_path)
        recorded = fields.get("package_manifest_sha256", "").strip()
        if recorded and not recorded.startswith("PENDING"):
            expected_hash = recorded.replace("sha256:", "")
            if expected_hash != manifest_sha256:
                failures.append("release_decision_record package_manifest_sha256 mismatch")

    if args.check_git_tag:
        tag = str(identity.get("git_tag") or f"v{expected_version}")
        try:
            tags = subprocess.check_output(["git", "tag", "-l", tag], text=True, cwd=str(repo_root)).strip()
            if not tags:
                failures.append(f"git tag {tag} not found")
        except Exception as exc:
            failures.append(f"git tag check failed: {exc}")

    if failures:
        raise SystemExit("package identity gate FAIL:\n" + "\n".join(failures))
    print(f"package identity gate PASS: {expected_id} {expected_version}")


if __name__ == "__main__":
    main()
