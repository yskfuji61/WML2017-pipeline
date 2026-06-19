"""E2E run context and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class E2ERunContext:
    repo_root: Path
    run_id: str
    files_root: str
    seed: int
    work_dir: Path
    config_path: Path
    sha256sums: str
    official_metrics: str
    skip_train: bool
    no_inspect_images: bool
    allow_dirty_git: bool
    max_epochs: int | None = None
    overwrite_run: bool = False


def validate_run_context(ctx: E2ERunContext) -> None:
    if not ctx.files_root.strip():
        raise ValueError("files_root is required")
    if not ctx.run_id.strip():
        raise ValueError("run_id is required")
    if ctx.seed < 0:
        raise ValueError("seed must be non-negative")
