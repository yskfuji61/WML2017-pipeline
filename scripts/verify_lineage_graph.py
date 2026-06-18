#!/usr/bin/env python3
"""Verify lineage graph completeness for a run (v2)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def _load_reviews(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify run lineage graph.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--require-artifact-hashes", action="store_true")
    parser.add_argument("--require-source-review", action="store_true")
    parser.add_argument("--require-release-decision", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = repo_root / "artifacts" / "runs" / args.run_id
    graph_path = run_dir / "lineage" / "lineage_graph.json"
    if not graph_path.exists():
        graph_path = run_dir / "lineage_graph.json"
    if not graph_path.exists():
        raise SystemExit(f"missing lineage graph: {graph_path}")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if graph.get("run_id") != args.run_id:
        failures.append(f"graph run_id mismatch: {graph.get('run_id')}")

    ctx_path = run_dir / "run_context.json"
    if ctx_path.exists():
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        if ctx.get("git_dirty") is True:
            failures.append("run_context git_dirty == true")
        if graph.get("code_commit") and ctx.get("code_commit") != graph.get("code_commit"):
            failures.append("code_commit mismatch between run_context and lineage graph")

    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    required_nodes = [
        "source:SRC-WMH-DATASET-OFFICIAL",
        "data:dataset_manifest",
        "audit:label_audit",
        "split:split_manifest",
        "config:train_config",
        "run:training",
        "model:checkpoint",
        "prediction:manifest",
        "eval:case_metrics",
        "release:decision",
    ]
    for node_id in required_nodes:
        if node_id not in nodes:
            failures.append(f"missing node: {node_id}")

    required_edges = [
        ("source:SRC-WMH-DATASET-OFFICIAL", "data:dataset_manifest"),
        ("data:dataset_manifest", "audit:label_audit"),
        ("audit:label_audit", "split:split_manifest"),
        ("split:split_manifest", "config:train_config"),
        ("config:train_config", "run:training"),
        ("run:training", "model:checkpoint"),
        ("model:checkpoint", "prediction:manifest"),
        ("prediction:manifest", "eval:case_metrics"),
        ("eval:case_metrics", "release:decision"),
    ]
    edge_set = {(e["from"], e["to"]) for e in graph.get("edges", [])}
    for frm, to in required_edges:
        if (frm, to) not in edge_set:
            failures.append(f"missing edge: {frm} -> {to}")

    if args.require_artifact_hashes:
        for node_id in [
            "data:dataset_manifest",
            "audit:label_audit",
            "split:split_manifest",
            "config:train_config",
            "model:checkpoint",
            "prediction:manifest",
            "eval:case_metrics",
        ]:
            if not nodes.get(node_id, {}).get("sha256"):
                failures.append(f"missing sha256 on node: {node_id}")

    if args.require_source_review:
        reviews = _load_reviews(repo_root / "registry/review_approval_register_wmh2017.csv")
        src = next((r for r in reviews if r.get("review_type") == "source_review"), None)
        if not src or src.get("status", "").upper() != "APPROVED":
            failures.append("source/license review not APPROVED")

    if args.require_release_decision:
        decision_path = repo_root / "docs" / "release_decisions" / f"release_decision_{args.run_id}.yaml"
        if not decision_path.exists():
            failures.append(f"missing release decision: {decision_path}")
        else:
            decision = yaml.safe_load(decision_path.read_text(encoding="utf-8")) or {}
            if "APPROVED" not in str(decision.get("decision", "")):
                failures.append(f"release decision not approved: {decision.get('decision')}")
            manifest_path = repo_root / "reports" / "full_package_manifest.json"
            if manifest_path.exists() and decision.get("package_manifest_sha256"):
                import hashlib

                h = hashlib.sha256()
                with manifest_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                actual = h.hexdigest()
                recorded = str(decision.get("package_manifest_sha256", "")).replace("sha256:", "")
                if recorded not in {"", "PENDING", "PENDING_GENERATED_MANIFEST"} and actual != recorded:
                    failures.append("release decision package_manifest_sha256 mismatch")

    if not (run_dir / "artifact_manifest.json").exists():
        failures.append("missing artifact_manifest.json")

    if failures:
        raise SystemExit("lineage graph gate FAIL:\n" + "\n".join(failures))
    print(f"lineage graph gate PASS: {graph_path}")


if __name__ == "__main__":
    main()
