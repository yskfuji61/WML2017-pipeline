#!/usr/bin/env python3
"""Environment doctor for WMH2017 local PoC runs.

Verifies that core scientific stack imports succeed without segfault/SIGABRT and
that installed package versions match requirements-lock.txt when present.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "nibabel",
    "scipy.ndimage",
    "torch",
    "monai",
)


def _parse_lock_versions() -> dict[str, str]:
    if not LOCK_FILE.exists():
        return {}
    versions: dict[str, str] = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==([0-9.]+)$", line)
        if match:
            versions[match.group(1).lower()] = match.group(2)
    return versions


def _import_probe(module: str) -> tuple[bool, str]:
    code = (
        "import importlib; "
        f"m=importlib.import_module({module!r}); "
        "v=getattr(m,'__version__',getattr(getattr(m,'version',None),'version','ok')); "
        "print(v)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return False, detail or f"import failed rc={completed.returncode}"
    version = (completed.stdout or "").strip() or "ok"
    return True, version


def _version_ok(module: str, observed: str, expected: dict[str, str]) -> tuple[bool, str]:
    key = module.split(".")[0].lower()
    expected_version = expected.get(key)
    if not expected_version or expected_version == "ok":
        return True, ""
    if observed.startswith(expected_version):
        return True, ""
    return False, f"expected {expected_version}, observed {observed}"


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    expected = _parse_lock_versions()
    failures: list[str] = []

    for module in REQUIRED_MODULES:
        ok, detail = _import_probe(module)
        if not ok:
            failures.append(f"{module}: FAIL ({detail})")
            print(f"{module}: FAIL ({detail})")
            continue
        version_ok, version_msg = _version_ok(module, detail, expected)
        if version_ok:
            print(f"{module}: OK ({detail})")
        else:
            failures.append(f"{module}: VERSION_MISMATCH ({version_msg})")
            print(f"{module}: VERSION_MISMATCH ({version_msg})")

    wmh_root = os.environ.get("WMH2017_ROOT", "")
    if wmh_root:
        root = Path(wmh_root).expanduser()
        if root.is_dir():
            print(f"WMH2017_ROOT: OK ({root})")
        else:
            failures.append(f"WMH2017_ROOT not a directory: {wmh_root}")
            print("WMH2017_ROOT: FAIL (not a directory)")
    else:
        print("WMH2017_ROOT: unset (set before e2e/smoke runs)")

    if failures:
        print("\nenvironment doctor FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nenvironment doctor PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
