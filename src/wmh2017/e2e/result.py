"""E2E stage and pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageResult:
    name: str
    status: str
    artifacts: list[str] = field(default_factory=list)


@dataclass
class E2EResult:
    work_dir: Path
    manifest_path: Path
    stage_status: dict[str, str]
    stage_results: list[StageResult] = field(default_factory=list)
