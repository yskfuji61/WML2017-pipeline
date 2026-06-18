"""Offline pipeline observability reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_run_observability(
    *,
    run_id: str,
    release_state: str = "NOT_READY_FOR_PREVIEW",
    dataset_summary: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    inference_summary: dict[str, Any] | None = None,
    evaluation_summary: dict[str, Any] | None = None,
    stage_status: dict[str, str] | None = None,
    alerts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "PASS",
        "release_state": release_state,
        "stage_status": stage_status or {},
        "run_health": {
            "dataset": dataset_summary or {},
            "training": training_summary or {},
            "inference": inference_summary or {},
            "evaluation": evaluation_summary or {},
        },
        "lineage_quality": {},
        "evaluation_quality": {},
        "security_quality": {},
        "alerts": alerts or [],
    }


def write_run_observability(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_offline_dashboard(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# Offline Run Dashboard", ""]
    for summary in summaries:
        lines.append(f"## {summary.get('run_id', 'unknown')}")
        lines.append(f"- status: {summary.get('status', 'UNKNOWN')}")
        for stage, status in (summary.get("stage_status") or {}).items():
            lines.append(f"- {stage}: {status}")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
