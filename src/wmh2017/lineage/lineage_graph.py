"""Lineage graph generation for WMH2017 run evidence (v2 schema)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wmh2017.lineage.hashes import sha256_path
from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown


def build_lineage_graph(
    *,
    run_id: str,
    artifact_hashes: dict[str, str],
    package_version: str = "0.2.3",
    release_decision_status: str = "NOT_APPROVED",
) -> dict[str, Any]:
    code_commit = git_commit_or_unknown()
    nodes = [
        {
            "id": "source:SRC-WMH-DATASET-OFFICIAL",
            "type": "source_register",
            "evidence": "registry/source_register_wmh2017.csv",
        },
        {
            "id": "data:dataset_manifest",
            "type": "dataset_manifest",
            "sha256": artifact_hashes.get("dataset_manifest", ""),
        },
        {
            "id": "audit:label_audit",
            "type": "label_audit",
            "sha256": artifact_hashes.get("label_audit", ""),
        },
        {
            "id": "split:split_manifest",
            "type": "split_manifest",
            "sha256": artifact_hashes.get("split_manifest", ""),
        },
        {
            "id": "config:train_config",
            "type": "config_snapshot",
            "sha256": artifact_hashes.get("train_config", ""),
        },
        {
            "id": "run:training",
            "type": "training_run",
            "run_id": run_id,
        },
        {
            "id": "model:checkpoint",
            "type": "model_artifact",
            "sha256": artifact_hashes.get("model", ""),
        },
        {
            "id": "prediction:manifest",
            "type": "prediction_manifest",
            "sha256": artifact_hashes.get("prediction_manifest", ""),
        },
        {
            "id": "eval:case_metrics",
            "type": "metric_table",
            "sha256": artifact_hashes.get("case_metrics", ""),
        },
        {
            "id": "release:decision",
            "type": "release_decision",
            "status": release_decision_status,
        },
    ]
    edges = [
        {"from": "source:SRC-WMH-DATASET-OFFICIAL", "to": "data:dataset_manifest"},
        {"from": "data:dataset_manifest", "to": "audit:label_audit"},
        {"from": "audit:label_audit", "to": "split:split_manifest"},
        {"from": "split:split_manifest", "to": "config:train_config"},
        {"from": "config:train_config", "to": "run:training"},
        {"from": "run:training", "to": "model:checkpoint"},
        {"from": "model:checkpoint", "to": "prediction:manifest"},
        {"from": "prediction:manifest", "to": "eval:case_metrics"},
        {"from": "eval:case_metrics", "to": "release:decision"},
    ]
    return {
        "lineage_id": f"lineage-{run_id}",
        "run_id": run_id,
        "package_version": package_version,
        "code_commit": code_commit,
        "nodes": nodes,
        "edges": edges,
    }


def artifact_hashes_from_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in data.get("artifacts", []):
        out[str(item.get("name", ""))] = str(item.get("sha256", ""))
    return out


def write_lineage_graph(path: Path, graph: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = sha256_path(path)
    sidecar = path.parent / "lineage_graph.sha256"
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def compute_evidence_binder_hash(binder_yaml_path: Path) -> str:
    if not binder_yaml_path.exists():
        return ""
    return sha256_path(binder_yaml_path)
