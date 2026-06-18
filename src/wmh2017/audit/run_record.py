"""Run evidence helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wmh2017.lineage.hashes import sha256_path
from wmh2017.lineage.hashes import write_json as _write_json
from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown, package_versions

write_json = _write_json


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


def append_run_manifest(row: dict[str, Any], path: str | Path = "registry/runs/run_manifest.csv") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False)
