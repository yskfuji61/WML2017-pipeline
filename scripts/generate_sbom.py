#!/usr/bin/env python3
"""Generate CycloneDX SBOM and license report from locked dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_lock_packages(lock_path: Path) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        if line.strip() == "":
            if current.get("name"):
                packages.append(current)
            current = {}
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if m:
            current = {"name": m.group(1), "version": m.group(2), "license": "UNKNOWN"}
    if current.get("name"):
        packages.append(current)
    return packages


def generate_cyclonedx(lock_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "cyclonedx-py",
                "requirements",
                str(lock_path),
                "-o",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # Fallback minimal CycloneDX when cyclonedx-py unavailable
        payload = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {"type": "library", "name": pkg["name"], "version": pkg["version"]}
                for pkg in parse_lock_packages(lock_path)
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return len(data.get("components", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBOM and license report.")
    parser.add_argument("--lock-file", default="requirements-lock.txt")
    parser.add_argument("--cdx-out", default="reports/security/sbom.cdx.json")
    parser.add_argument("--license-out", default="reports/security/license_report.json")
    parser.add_argument("--package-version", default="0.2.3")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / args.lock_file
    if not lock_path.exists():
        raise SystemExit(f"lock file not found: {lock_path}")

    packages = parse_lock_packages(lock_path)
    lock_hash = sha256_file(lock_path)
    cdx_out = repo_root / args.cdx_out
    component_count = generate_cyclonedx(lock_path, cdx_out)

    license_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_file": args.lock_file,
        "lock_file_sha256": lock_hash,
        "package_count": len(packages),
        "sbom_component_count": component_count,
        "packages": packages,
    }
    license_out = repo_root / args.license_out
    license_out.write_text(json.dumps(license_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote CycloneDX SBOM: {cdx_out} ({component_count} components)")
    print(f"Wrote license report: {license_out}")


if __name__ == "__main__":
    main()
