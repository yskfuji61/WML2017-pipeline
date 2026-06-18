#!/usr/bin/env python3
"""Verify lineage graph completeness for a run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify run lineage graph.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--require-artifact-hashes", action="store_true")
    parser.add_argument("--require-source-review", action="store_true")
    parser.add_argument("--require-release-decision", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = repo_root / "artifacts" / "runs" / args.run_id
    graph_path = run_dir / "lineage_graph.json"
    if not graph_path.exists():
        raise SystemExit(f"missing lineage graph: {graph_path}")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    required_nodes = [
        "source:SRC-WMH-DATASET-OFFICIAL",
        "data:dataset_manifest",
        "split:split_manifest",
        "config:train_config",
        "model:checkpoint",
        "eval:case_metrics",
        "release:decision",
    ]
    for node_id in required_nodes:
        if node_id not in nodes:
            failures.append(f"missing node: {node_id}")

    if args.require_artifact_hashes:
        for node_id in ["data:dataset_manifest", "split:split_manifest", "config:train_config", "model:checkpoint", "eval:case_metrics"]:
            node = nodes.get(node_id, {})
            if not node.get("sha256"):
                failures.append(f"missing sha256 on node: {node_id}")

    if args.require_release_decision:
        decision = nodes.get("release:decision", {})
        if decision.get("status") == "NOT_APPROVED":
            failures.append("release decision status NOT_APPROVED")

    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        failures.append("missing artifact_manifest.json")

    ctx_path = run_dir / "run_context.json"
    if ctx_path.exists():
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        if ctx.get("git_dirty") is True:
            failures.append("run_context git_dirty == true")

    if failures:
        raise SystemExit("lineage graph gate FAIL:\n" + "\n".join(failures))
    print(f"lineage graph gate PASS: {graph_path}")


if __name__ == "__main__":
    main()
