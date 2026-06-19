#!/usr/bin/env python3
"""Environment doctor for WMH2017 local PoC runs.

Verifies that core scientific stack imports succeed without segfault/SIGABRT and
that installed package versions match requirements-lock.txt when present.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"
IMPORT_SMOKE_JSON = REPO_ROOT / "reports/env/import_smoke.json"

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "nibabel",
    "scipy.ndimage",
    "torch",
    "monai",
)

DOC_PLACEHOLDER_PATTERN = re.compile(r"^<[^>]+>$|LOCAL_WMH2017_FILES_ROOT", re.IGNORECASE)


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


def _distribution_name(module: str) -> str:
    return module.split(".")[0].lower()


def _is_doc_placeholder(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    return bool(DOC_PLACEHOLDER_PATTERN.search(stripped))


def _import_probe(module: str) -> tuple[bool, str]:
    dist_name = _distribution_name(module)
    code = (
        "import importlib, importlib.metadata; "
        f"importlib.import_module({module!r}); "
        f"print(importlib.metadata.version({dist_name!r}))"
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
    key = _distribution_name(module)
    expected_version = expected.get(key)
    if not expected_version or expected_version == "ok":
        return True, ""
    if observed.startswith(expected_version):
        return True, ""
    return False, f"expected {expected_version}, observed {observed}"


def _check_wmh2017_root(raw: str) -> tuple[str, bool, str | None]:
    """Return (status_label, is_valid_dir, warning_message)."""
    if not raw or _is_doc_placeholder(raw):
        return "unset", False, None
    root = Path(raw).expanduser()
    if root.is_dir():
        return "ok", True, None
    return "invalid", False, f"WMH2017_ROOT not a directory: {raw}"


def _mps_available() -> bool | None:
    ok, _ = _import_probe("torch")
    if not ok:
        return None
    code = (
        "import torch; "
        "print('true' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'false')"
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
        return None
    value = (completed.stdout or "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _write_import_smoke(payload: dict) -> None:
    IMPORT_SMOKE_JSON.parent.mkdir(parents=True, exist_ok=True)
    IMPORT_SMOKE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    expected = _parse_lock_versions()
    import_failures: list[str] = []
    warnings: list[str] = []
    module_results: dict[str, dict[str, str | bool]] = {}

    for module in REQUIRED_MODULES:
        ok, detail = _import_probe(module)
        version_ok, version_msg = _version_ok(module, detail, expected) if ok else (False, detail)
        module_results[module] = {
            "ok": ok and version_ok,
            "detail": detail if ok else detail,
            "version_match": version_ok if ok else False,
        }
        if not ok:
            import_failures.append(f"{module}: FAIL ({detail})")
            print(f"{module}: FAIL ({detail})")
            continue
        if version_ok:
            print(f"{module}: OK ({detail})")
        else:
            import_failures.append(f"{module}: VERSION_MISMATCH ({version_msg})")
            print(f"{module}: VERSION_MISMATCH ({version_msg})")

    mps_available = _mps_available()
    if mps_available is True:
        print("mps_available: true")
    elif mps_available is False:
        print("mps_available: false")
    else:
        print("mps_available: unknown")

    wmh_root = os.environ.get("WMH2017_ROOT", "")
    wmh_status, wmh_root_ok, wmh_warning = _check_wmh2017_root(wmh_root)
    if wmh_status == "ok":
        root = Path(wmh_root).expanduser()
        print(f"WMH2017_ROOT: OK ({root})")
    elif wmh_status == "unset":
        print("WMH2017_ROOT: unset (set before e2e/smoke runs)")
    else:
        print("WMH2017_ROOT: WARN (invalid path; required before e2e/smoke runs)")
        if wmh_warning:
            warnings.append(wmh_warning)

    imports_ok = not import_failures
    payload = {
        "checked_at_utc": datetime.now(tz=UTC).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "imports_ok": imports_ok,
        "modules": module_results,
        "mps_available": mps_available,
        "wmh2017_root_set": bool(wmh_root) and not _is_doc_placeholder(wmh_root),
        "wmh2017_root_ok": wmh_root_ok,
        "wmh2017_root_status": wmh_status,
        "import_failures": import_failures,
        "warnings": warnings,
        "failures": import_failures,
    }
    _write_import_smoke(payload)
    print(f"\nWrote import smoke report: {IMPORT_SMOKE_JSON}")

    if warnings:
        print("\nenvironment doctor WARN")
        for item in warnings:
            print(f"  - {item}")

    if import_failures:
        print("\nenvironment doctor FAIL")
        for item in import_failures:
            print(f"  - {item}")
        return 1

    print("\nenvironment doctor PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
