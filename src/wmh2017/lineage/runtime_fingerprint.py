"""Capture runtime environment fingerprint for reproducibility audits."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from wmh2017.lineage.hashes import sha256_path


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def git_commit_or_unknown() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["torch", "monai", "numpy", "pandas", "scipy", "nibabel"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_installed"
    return versions


def build_runtime_fingerprint(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    lock = root / "requirements-lock.txt"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit_or_unknown(),
        "git_dirty": git_dirty(),
        "dependency_lock_path": str(lock),
        "dependency_lock_hash": sha256_path(lock),
        "package_versions": package_versions(),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_runtime_fingerprint(out_path: str | Path, *, repo_root: str | Path = ".") -> dict[str, Any]:
    payload = build_runtime_fingerprint(repo_root=repo_root)
    write_json(out_path, payload)
    return payload
