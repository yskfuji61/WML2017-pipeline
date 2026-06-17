"""Run evidence helpers."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    if not str(path) or not p.exists() or p.is_dir():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def make_run_row(
    run_id: str,
    run_purpose: str,
    config_path: str,
    dataset_manifest: str,
    split_manifest: str,
    model_name: str = "MONAI UNet",
    model_version: str = "smoke",
    seed: int = 42,
    device: str = "auto",
    status: str = "created",
    metric_json_path: str = "",
    overlay_dir: str = "reports/overlays",
    checkpoint_path: str = "",
    prediction_dir: str = "",
    notes: str = "",
) -> dict[str, Any]:
    versions = package_versions()
    return {
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": git_commit_or_unknown(),
        "config_path": config_path,
        "config_hash": sha256_path(config_path),
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_hash": sha256_path(dataset_manifest),
        "split_manifest": split_manifest,
        "split_manifest_hash": sha256_path(split_manifest),
        "model_name": model_name,
        "model_version": model_version,
        "monai_version": versions.get("monai", "not_installed"),
        "pytorch_version": versions.get("torch", "not_installed"),
        "seed": seed,
        "device": device,
        "status": status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": sha256_path(checkpoint_path),
        "prediction_dir": prediction_dir,
        "metric_json_path": metric_json_path,
        "metric_json_hash": sha256_path(metric_json_path),
        "overlay_dir": overlay_dir,
        "notes": notes,
    }


def append_run_manifest(row: dict[str, Any], path: str | Path = "registry/run_manifest.csv") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)
