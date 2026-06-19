"""Lightweight JSON schema validation without external jsonschema dependency."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


def load_schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_required_keys(payload: dict, schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    required = schema.get("required", [])
    failures: list[str] = []
    for key in required:
        if key not in payload:
            failures.append(f"missing required key: {key}")
    return failures


def csv_header_from_schema(schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    return list(schema.get("required", []))
