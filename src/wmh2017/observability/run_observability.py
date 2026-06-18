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
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "release_state": release_state,
        "dataset": dataset_summary or {},
        "training": training_summary or {},
        "inference": inference_summary or {},
        "evaluation": evaluation_summary or {},
        "alerts": [],
    }


def write_run_observability(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
