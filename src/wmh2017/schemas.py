"""v4 dataset manifest schema helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_POLICY_PENDING = "LABEL_POLICY_PENDING"


def redact_path(path: str) -> str:
    if not path or path in {"nan", "None"}:
        return "REDACTED_OR_LOCAL_ONLY_OR_NULL"
    return "REDACTED_OR_LOCAL_ONLY"


def manifest_json_from_csv(df: pd.DataFrame, *, root: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        flair_path = str(row.get("flair_pre_path") or row.get("flair_path") or "")
        t1_path = str(row.get("t1_pre_path") or row.get("t1_path") or "")
        label_path = str(row.get("wmh_path") or row.get("mask_path") or "")
        shape = _parse_shape(row.get("flair_pre_shape") or row.get("flair_shape"))
        spacing = _parse_spacing(row.get("flair_pre_spacing") or row.get("flair_spacing"))
        label_values = _parse_label_values(row.get("label_values"))
        cases.append(
            {
                "case_id": str(row.get("case_id", "")),
                "site": str(row.get("site", "UNSPECIFIED")),
                "modalities": {
                    "flair": {
                        "path": redact_path(flair_path),
                        "sha256": str(row.get("flair_pre_sha256") or row.get("flair_sha256") or ""),
                        "shape": shape,
                        "spacing": spacing,
                        "orientation": "PENDING_CONFIRMATION",
                    },
                    "t1": {
                        "path": redact_path(t1_path) if t1_path else "REDACTED_OR_LOCAL_ONLY_OR_NULL",
                        "sha256": str(row.get("t1_pre_sha256") or row.get("t1_sha256") or "") or None,
                    },
                },
                "label": {
                    "path": redact_path(label_path),
                    "sha256": str(row.get("wmh_sha256") or row.get("mask_sha256") or ""),
                    "label_values": label_values,
                    "label_policy": LABEL_POLICY_PENDING,
                },
            }
        )
    payload = {
        "dataset_id": "wmh2017_public_poc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "PENDING_CONFIRMATION",
        "root": "REDACTED_OR_LOCAL_ONLY",
        "source_id": "PENDING_CONFIRMATION",
        "license_or_terms": "PENDING_CONFIRMATION",
        "dlp_class": "PUBLIC_CHALLENGE_DATA",
        "cases": cases,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _parse_shape(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [0, 0, 0]
    text = str(value).strip()
    if not text:
        return [0, 0, 0]
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    try:
        return [int(float(p)) for p in parts[:3]] + [0] * max(0, 3 - len(parts))
    except ValueError:
        return [0, 0, 0]


def _parse_spacing(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [0.0, 0.0, 0.0]
    text = str(value).strip()
    if not text:
        return [0.0, 0.0, 0.0]
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts[:3]]
        return vals + [0.0] * max(0, 3 - len(vals))
    except ValueError:
        return [0.0, 0.0, 0.0]


def _parse_label_values(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(float(x)) for x in text.replace("[", "").replace("]", "").split(",") if x.strip()]


def write_manifest_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
