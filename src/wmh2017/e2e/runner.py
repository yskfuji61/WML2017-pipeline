"""E2E pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from wmh2017.e2e.context import E2ERunContext, validate_run_context
from wmh2017.e2e.result import E2EResult, StageResult
from wmh2017.e2e.stages import (
    PipelineState,
    audit_labels_stage,
    create_or_load_split_stage,
    evaluate_stage,
    prepare_dataset_stage,
    train_smoke_model_stage,
)
from wmh2017.lineage.artifact_manifest import ArtifactManifest
from wmh2017.lineage.hashes import write_hash_sidecar
from wmh2017.lineage.lineage_graph import (
    artifact_hashes_from_manifest,
    build_lineage_graph,
    write_lineage_graph,
)
from wmh2017.lineage.run_context import init_run_directory
from wmh2017.lineage.runtime_fingerprint import git_dirty, write_runtime_fingerprint
from wmh2017.observability.event_log import (
    EVENT_ARTIFACT_HASHED,
    EVENT_DATASET_AUDIT_STARTED,
    EVENT_EVALUATION_COMPLETED,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_FAILED,
    EVENT_RUN_STARTED,
    EVENT_SPLIT_CREATED,
    EVENT_TRAINING_COMPLETED,
    EVENT_TRAINING_STARTED,
    emit_event,
)
from wmh2017.observability.run_observability import build_run_observability, write_run_observability


def _load_security_quality(repo_root: Path) -> dict:
    policy_path = repo_root / "reports/security/security_policy_result.json"
    if not policy_path.exists():
        return {}
    import json

    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    return {"status": payload.get("status", "UNKNOWN"), "path": str(policy_path)}


def write_artifact_manifest_stage(state: PipelineState) -> StageResult:
    manifest_path = state.work_dir / "artifact_manifest.json"
    state.manifest.write(manifest_path)
    write_hash_sidecar(manifest_path)
    return state._record("artifact_manifest", "PASS", [str(manifest_path)])


def write_lineage_stage(state: PipelineState) -> StageResult:
    manifest_path = state.work_dir / "artifact_manifest.json"
    graph = build_lineage_graph(
        run_id=state.ctx.run_id,
        artifact_hashes=artifact_hashes_from_manifest(manifest_path),
        package_version="0.2.3",
    )
    lineage_path = state.work_dir / "lineage" / "lineage_graph.json"
    write_lineage_graph(lineage_path, graph)
    return state._record("lineage", "PASS", [str(lineage_path)])


def write_evidence_stage(state: PipelineState) -> StageResult:
    stage_status = dict(state.stage_status)
    if "split" in stage_status:
        stage_status["split_generation"] = stage_status.pop("split")
    stage_status.setdefault("split_generation", "SKIP")
    if "prediction" in stage_status:
        stage_status["prediction_export"] = stage_status.pop("prediction")
    else:
        stage_status.setdefault("prediction_export", "SKIP")
    stage_status["lineage_verification"] = "PENDING"
    stage_status["binder_verification"] = "PENDING"

    observability = build_run_observability(
        run_id=state.ctx.run_id,
        release_state="PREVIEW_CANDIDATE",
        dataset_summary={"status": stage_status.get("dataset_audit", "SKIP")},
        training_summary={"status": stage_status.get("training", "SKIP")},
        inference_summary={"status": stage_status.get("prediction_export", "SKIP")},
        evaluation_summary={"status": stage_status.get("evaluation", "SKIP")},
        stage_status=stage_status,
    )
    observability["status"] = "PASS"
    observability["security_quality"] = _load_security_quality(state.repo_root)
    out_path = state.work_dir / "observability/offline_run_summary.json"
    write_run_observability(out_path, observability)
    state.stage_status.update(stage_status)
    return state._record("evidence", "PASS", [str(out_path)])


def run_pipeline(ctx: E2ERunContext) -> E2EResult:
    validate_run_context(ctx)
    events: list[dict] = [emit_event(EVENT_RUN_STARTED, run_id=ctx.run_id)]
    try:
        init_run_directory(ctx.work_dir, run_id=ctx.run_id, wmh2017_root=ctx.files_root, seed=ctx.seed)
        write_runtime_fingerprint(ctx.work_dir / "runtime_fingerprint.json", repo_root=ctx.repo_root)

        if git_dirty() and not ctx.allow_dirty_git:
            raise SystemExit("git working tree is dirty; commit changes or pass --allow-dirty-git")

        state = PipelineState(ctx=ctx, manifest=ArtifactManifest(ctx.run_id))
        events.append(emit_event(EVENT_DATASET_AUDIT_STARTED, run_id=ctx.run_id))
        prepare_dataset_stage(state)
        audit_labels_stage(state)
        create_or_load_split_stage(state)
        events.append(emit_event(EVENT_SPLIT_CREATED, run_id=ctx.run_id))
        if not ctx.skip_train:
            events.append(emit_event(EVENT_TRAINING_STARTED, run_id=ctx.run_id))
        train_smoke_model_stage(state)
        if state.stage_status.get("training") == "PASS":
            events.append(emit_event(EVENT_TRAINING_COMPLETED, run_id=ctx.run_id))
        evaluate_stage(state)
        if state.stage_status.get("evaluation") == "PASS":
            events.append(emit_event(EVENT_EVALUATION_COMPLETED, run_id=ctx.run_id))
        write_artifact_manifest_stage(state)
        events.append(emit_event(EVENT_ARTIFACT_HASHED, run_id=ctx.run_id))
        write_lineage_stage(state)
        write_evidence_stage(state)
        events.append(emit_event(EVENT_RUN_COMPLETED, run_id=ctx.run_id))
    except SystemExit as exc:
        events.append(emit_event(EVENT_RUN_FAILED, run_id=ctx.run_id, payload={"reason": str(exc)}))
        raise

    manifest_path = ctx.work_dir / "artifact_manifest.json"
    event_log_path = ctx.work_dir / "observability/event_log.jsonl"
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    event_log_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in events) + "\n",
        encoding="utf-8",
    )
    return E2EResult(
        work_dir=ctx.work_dir,
        manifest_path=manifest_path,
        stage_status=state.stage_status,
        stage_results=state.stage_results,
    )
