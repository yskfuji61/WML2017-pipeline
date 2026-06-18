"""Run context initialization for offline pipeline runs."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown, git_dirty, package_versions
from wmh2017.security.path_redaction import redact_path


def build_run_context(
    *,
    run_id: str,
    wmh2017_root: str | Path = "",
    seed: int = 42,
    device: str = "auto",
    owner: str = "research-dev",
    release_state: str = "PREVIEW_CANDIDATE",
    package_version: str = "0.0.0.0",
    config_hash: str = "",
    dataset_manifest_hash: str = "",
    split_manifest_hash: str = "",
) -> dict[str, Any]:
    versions = package_versions()
    return {
        "run_id": run_id,
        "package_version": package_version,
        "code_commit": git_commit_or_unknown(),
        "git_dirty": git_dirty(),
        "owner": owner,
        "release_state": release_state,
        "python_version": versions.get("python", ""),
        "platform": versions.get("platform", ""),
        "torch_version": versions.get("torch", "not_installed"),
        "monai_version": versions.get("monai", "not_installed"),
        "seed": seed,
        "device": device,
        "wmh2017_root_redacted": True,
        "wmh2017_root": redact_path(wmh2017_root),
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_git_state() -> dict[str, Any]:
    branch = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return {
        "commit": git_commit_or_unknown(),
        "branch": branch,
        "dirty": git_dirty(),
    }


def init_run_directory(
    run_dir: str | Path,
    *,
    run_id: str,
    wmh2017_root: str | Path = "",
    seed: int = 42,
    device: str = "auto",
    fail_if_exists: bool = True,
) -> Path:
    root = Path(run_dir)
    if fail_if_exists and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run directory already exists and is non-empty: {root}")

    subdirs = [
        "dataset",
        "label_audit",
        "splits",
        "configs",
        "logs",
        "checkpoints",
        "predictions",
        "evaluation",
        "observability",
        "release",
    ]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)

    ctx = build_run_context(run_id=run_id, wmh2017_root=wmh2017_root, seed=seed, device=device)
    (root / "run_context.json").write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "git_state.json").write_text(
        json.dumps(build_git_state(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return root


def update_run_context(run_dir: Path, **fields: Any) -> dict[str, Any]:
    ctx_path = run_dir / "run_context.json"
    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    ctx.update(fields)
    ctx_path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    return ctx


def append_command_log(run_dir: str | Path, step: dict[str, Any]) -> None:
    log_path = Path(run_dir) / "command_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")
