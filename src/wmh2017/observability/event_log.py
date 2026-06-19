"""Stable event names for WMH2017 pipeline observability."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

EVENT_RUN_STARTED = "wmh2017.run.started"
EVENT_RUN_COMPLETED = "wmh2017.run.completed"
EVENT_RUN_FAILED = "wmh2017.run.failed"
EVENT_DATASET_AUDIT_STARTED = "wmh2017.dataset.audit.started"
EVENT_DATASET_AUDIT_FAILED = "wmh2017.dataset.audit.failed"
EVENT_SPLIT_CREATED = "wmh2017.split.created"
EVENT_TRAINING_STARTED = "wmh2017.training.started"
EVENT_TRAINING_COMPLETED = "wmh2017.training.completed"
EVENT_EVALUATION_COMPLETED = "wmh2017.evaluation.completed"
EVENT_ARTIFACT_HASHED = "wmh2017.artifact.hashed"
EVENT_EVIDENCE_VERIFIED = "wmh2017.evidence.verified"


def emit_event(name: str, *, run_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    event = {
        "event": name,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if payload:
        event.update(payload)
    return event
