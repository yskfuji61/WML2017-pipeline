"""Artifact manifest DAG for run lineage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wmh2017.lineage.hashes import sha256_path, write_hash_sidecar


class ArtifactManifest:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.artifacts: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        path: str | Path,
        *,
        producer: str,
        inputs: list[str] | None = None,
        write_sidecar: bool = True,
    ) -> None:
        p = Path(path)
        digest = write_hash_sidecar(p) if write_sidecar and p.is_file() else sha256_path(p)
        self.artifacts.append(
            {
                "name": name,
                "path": p.as_posix(),
                "sha256": digest,
                "producer": producer,
                "inputs": inputs or [],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "artifacts": self.artifacts}

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
