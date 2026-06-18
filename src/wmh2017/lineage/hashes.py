"""Hash utilities for run artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    if not str(path) or not p.exists() or p.is_dir():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_hash_sidecar(path: str | Path) -> str:
    p = Path(path)
    digest = sha256_path(p)
    if digest:
        p.with_suffix(p.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
