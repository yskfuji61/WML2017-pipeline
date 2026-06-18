#!/usr/bin/env python3
"""Generate SPDX SBOM and license report from locked dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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


def build_spdx(packages: list[dict[str, str]], *, lock_hash: str, package_version: str) -> dict:
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "wmh2017-pipeline",
        "documentNamespace": f"https://github.com/yskfuji61/WML2017-pipeline/spdx/{package_version}",
        "creationInfo": {
            "created": datetime.now(timezone.utc).isoformat(),
            "creators": ["Tool: generate_sbom.py"],
        },
        "packages": [
            {
                "SPDXID": f"SPDXRef-Package-{i}",
                "name": pkg["name"],
                "versionInfo": pkg["version"],
                "licenseConcluded": pkg.get("license", "NOASSERTION"),
                "downloadLocation": "NOASSERTION",
            }
            for i, pkg in enumerate(packages, start=1)
        ],
        "externalRefs": [
            {
                "referenceCategory": "SECURITY",
                "referenceType": "sha256",
                "referenceLocator": f"requirements-lock.txt:{lock_hash}",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBOM and license report.")
    parser.add_argument("--lock-file", default="requirements-lock.txt")
    parser.add_argument("--out", default="reports/security/sbom.spdx.json")
    parser.add_argument("--license-out", default="reports/security/license_report.json")
    parser.add_argument("--package-version", default="0.0.0.0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / args.lock_file
    if not lock_path.exists():
        raise SystemExit(f"lock file not found: {lock_path}")

    packages = parse_lock_packages(lock_path)
    lock_hash = sha256_file(lock_path)
    sbom = build_spdx(packages, lock_hash=lock_hash, package_version=args.package_version)

    out = repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sbom, indent=2, ensure_ascii=False), encoding="utf-8")

    license_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_file": args.lock_file,
        "lock_file_sha256": lock_hash,
        "package_count": len(packages),
        "packages": packages,
        "note": "License fields are NOASSERTION until pip-licenses review completes.",
    }
    license_out = repo_root / args.license_out
    license_out.write_text(json.dumps(license_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote SBOM: {out}")
    print(f"Wrote license report: {license_out}")


if __name__ == "__main__":
    main()
