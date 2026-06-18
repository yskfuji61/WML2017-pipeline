"""Run context initialization for offline pipeline runs."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown, git_dirty
from wmh2017.security.path_redaction import redact_path


def build_run_context(
    *,
    run_id: str,
    wmh2017_root: str | Path = "",
    seed: int = 42,
    device: str = "auto",
    owner: str = "research-dev",
    release_state: str = "NOT_READY_FOR_PREVIEW",
    package_version: str = "0.2.2",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "code_commit": git_commit_or_unknown(),
        "git_dirty": git_dirty(),
        "package_version": package_version,
        "seed": seed,
        "device": device,
        "wmh2017_root": redact_path(wmh2017_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner": owner,
        "release_state": release_state,
    }


def build_git_state() -> dict[str, Any]:
    branch = "unknown"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
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
) -> Path:
    root = Path(run_dir)
    for sub in ["model", "predictions", "evaluation/local", "evaluation/official_parity", "observability", "release"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    ctx = build_run_context(run_id=run_id, wmh2017_root=wmh2017_root, seed=seed, device=device)
    (root / "run_context.json").write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "git_state.json").write_text(
        json.dumps(build_git_state(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return root


def append_command_log(run_dir: str | Path, step: dict[str, Any]) -> None:
    log_path = Path(run_dir) / "command_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")
