"""E2E pipeline stage implementations."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from wmh2017.e2e.context import E2ERunContext
from wmh2017.e2e.result import StageResult
from wmh2017.evaluation.metric_schema import validate_case_metrics_columns
from wmh2017.lineage.artifact_manifest import ArtifactManifest
from wmh2017.lineage.hashes import sha256_path, write_hash_sidecar, write_json
from wmh2017.lineage.prediction_manifest import (
    build_prediction_manifest,
    write_prediction_label_linkage,
    write_prediction_manifest,
)
from wmh2017.lineage.run_context import append_command_log, update_run_context


def run_command(cmd: Sequence[str], *, cwd: Path) -> dict:
    completed = subprocess.run(list(cmd), cwd=str(cwd), text=True, capture_output=True, check=False)
    return {
        "cmd": list(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def require_ok(step: dict) -> None:
    if int(step["returncode"]) != 0:
        raise SystemExit(
            "command failed:\n"
            + " ".join(step["cmd"])
            + "\nSTDOUT:\n"
            + step["stdout"]
            + "\nSTDERR:\n"
            + step["stderr"]
        )


def copy_to_nested(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    elif not dst.exists():
        raise FileNotFoundError(src)
    write_hash_sidecar(dst)


@dataclass
class PipelineState:
    ctx: E2ERunContext
    manifest: ArtifactManifest
    stage_status: dict[str, str] = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    manifest_csv: Path | None = None
    manifest_json: Path | None = None
    split_csv: Path | None = None
    split_manifest: Path | None = None
    train_config: Path | None = None
    model_dst: Path | None = None

    @property
    def work_dir(self) -> Path:
        return self.ctx.work_dir

    @property
    def repo_root(self) -> Path:
        return self.ctx.repo_root

    def _record(self, name: str, status: str, artifacts: list[str] | None = None) -> StageResult:
        result = StageResult(name=name, status=status, artifacts=artifacts or [])
        self.stage_results.append(result)
        self.stage_status[name] = status
        return result


def prepare_dataset_stage(state: PipelineState) -> StageResult:
    ctx = state.ctx
    dataset_dir = ctx.work_dir / "dataset"
    manifest_csv = dataset_dir / "dataset_manifest.csv"
    manifest_json = dataset_dir / "dataset_manifest.json"
    manifest_summary = dataset_dir / "dataset_manifest.summary.json"
    dataset_cmd = [
        sys.executable,
        "scripts/audit_wmh2017_dataset.py",
        "--root",
        ctx.files_root,
        "--out",
        str(manifest_csv),
        "--summary-out",
        str(manifest_summary),
        "--strict-counts",
        "--sha256sums",
        ctx.sha256sums,
    ]
    if not ctx.no_inspect_images:
        dataset_cmd.extend(["--inspect-images", "--fail-on-metadata-error"])
    step = run_command(dataset_cmd, cwd=ctx.repo_root)
    append_command_log(ctx.work_dir, step)
    require_ok(step)
    copy_to_nested(manifest_csv, manifest_json)
    state.manifest.add("dataset_manifest", manifest_json, producer="scripts/audit_wmh2017_dataset.py")
    state.manifest_csv = manifest_csv
    state.manifest_json = manifest_json
    return state._record("dataset_audit", "PASS", [str(manifest_json)])


def audit_labels_stage(state: PipelineState) -> StageResult:
    ctx = state.ctx
    assert state.manifest_csv is not None
    label_dir = ctx.work_dir / "label_audit"
    label_csv = label_dir / "label_audit.csv"
    label_json = label_dir / "label_audit.json"
    label_summary = label_dir / "label_audit.summary.json"
    label_cmd = [
        sys.executable,
        "scripts/audit_wmh2017_labels.py",
        "--manifest",
        str(state.manifest_csv),
        "--out",
        str(label_csv),
        "--summary-out",
        str(label_summary),
        "--split",
        "training",
        "--include-geometry",
        "--fail-on-error",
    ]
    step = run_command(label_cmd, cwd=ctx.repo_root)
    append_command_log(ctx.work_dir, step)
    require_ok(step)
    copy_to_nested(label_csv, label_json)
    state.manifest.add(
        "label_audit",
        label_json,
        producer="scripts/audit_wmh2017_labels.py",
        inputs=["dataset_manifest"],
    )
    return state._record("label_audit", "PASS", [str(label_json)])


def create_or_load_split_stage(state: PipelineState) -> StageResult:
    ctx = state.ctx
    assert state.manifest_csv is not None
    splits_dir = ctx.work_dir / "splits"
    split_cmd = [
        sys.executable,
        "scripts/make_wmh2017_splits.py",
        "--manifest",
        str(state.manifest_csv),
        "--seed",
        str(ctx.seed),
        "--out-dir",
        str(splits_dir),
    ]
    step = run_command(split_cmd, cwd=ctx.repo_root)
    append_command_log(ctx.work_dir, step)
    require_ok(step)
    split_csv = splits_dir / f"wmh2017_train_val_seed{ctx.seed}.csv"
    split_manifest = splits_dir / "split_manifest.json"
    split_manifest.write_text(split_csv.read_text(encoding="utf-8"), encoding="utf-8")
    write_hash_sidecar(split_manifest)
    state.manifest.add(
        "split_manifest",
        split_manifest,
        producer="scripts/make_wmh2017_splits.py",
        inputs=["dataset_manifest"],
    )
    assert state.manifest_json is not None
    update_run_context(
        ctx.work_dir,
        dataset_manifest_hash=sha256_path(state.manifest_json),
        split_manifest_hash=sha256_path(split_manifest),
    )
    state.split_csv = split_csv
    state.split_manifest = split_manifest
    return state._record("split", "PASS", [str(split_manifest)])


def train_smoke_model_stage(state: PipelineState) -> StageResult | None:
    if state.ctx.skip_train:
        return None
    ctx = state.ctx
    assert state.manifest_csv is not None and state.split_csv is not None
    configs_dir = ctx.work_dir / "configs"
    logs_dir = ctx.work_dir / "logs"
    ckpt_dir = ctx.work_dir / "checkpoints"
    train_config = configs_dir / "train_config.materialized.yaml"
    eval_config = configs_dir / "eval_config.materialized.yaml"
    pred_dir = ctx.work_dir / "predictions"
    cfg = yaml.safe_load(ctx.config_path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("run", {})
    cfg.setdefault("data", {})
    cfg["run"].update(
        {
            "run_id": ctx.run_id,
            "seed": ctx.seed,
            "output_dir": ctx.work_dir.as_posix(),
            "run_manifest": "registry/runs/run_manifest.csv",
        }
    )
    cfg["data"].update(
        {
            "dataset_manifest": state.manifest_csv.as_posix(),
            "split_manifest": state.split_csv.as_posix(),
        }
    )
    if ctx.max_epochs is not None:
        cfg.setdefault("training", {})
        cfg["training"]["max_epochs"] = int(ctx.max_epochs)
    train_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_hash_sidecar(train_config)
    eval_cfg = {
        "run_id": ctx.run_id,
        "threshold": 0.5,
        "assigned_split": "val",
        "manifest_csv": state.manifest_csv.as_posix(),
        "split_csv": state.split_csv.as_posix(),
        "prediction_dir": pred_dir.as_posix(),
    }
    eval_config.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")
    write_hash_sidecar(eval_config)
    config_snapshot = configs_dir / "config_snapshot.sha256"
    combined = sha256_path(train_config) + sha256_path(eval_config)
    config_snapshot.write_text(combined + "\n", encoding="utf-8")
    state.manifest.add(
        "train_config",
        train_config,
        producer="scripts/run_wmh2017_e2e.py",
        inputs=["dataset_manifest", "split_manifest"],
    )
    update_run_context(
        ctx.work_dir,
        config_hash=sha256_path(train_config),
        config_snapshot_sha256=config_snapshot.read_text(encoding="utf-8").strip(),
    )
    train_cmd = [sys.executable, "scripts/train_wmh2017.py", "--config", str(train_config)]
    step = run_command(train_cmd, cwd=ctx.repo_root)
    append_command_log(ctx.work_dir, step)
    require_ok(step)
    state.stage_status["training"] = "PASS"

    model_src = ctx.work_dir / "checkpoints" / "model_smoke.pt"
    model_dst = ckpt_dir / "model.pt"
    if model_src.exists():
        copy_to_nested(model_src, model_dst)
        state.manifest.add(
            "model",
            model_dst,
            producer="src/wmh2017/training/train_monai.py",
            inputs=["split_manifest", "train_config"],
        )
        model_card = ckpt_dir / "model_card_fragment.json"
        write_json(
            model_card,
            {
                "run_id": ctx.run_id,
                "model_artifact_sha256": sha256_path(model_dst),
                "claim_boundary": "smoke training only; no performance claim",
            },
        )
    train_log_src = ctx.work_dir / "logs" / "train_log.jsonl"
    if train_log_src.exists():
        copy_to_nested(train_log_src, logs_dir / "train_log.jsonl")
    state.train_config = train_config
    state.model_dst = model_dst
    return state._record("training", "PASS", [str(train_config)])


def evaluate_stage(state: PipelineState) -> StageResult | None:
    if state.ctx.skip_train:
        return None
    ctx = state.ctx
    assert state.manifest_csv is not None and state.split_csv is not None
    assert state.train_config is not None
    pred_dir = ctx.work_dir / "predictions"
    eval_dir = ctx.work_dir / "evaluation"
    logs_dir = ctx.work_dir / "logs"
    model_dst = state.model_dst or (ctx.work_dir / "checkpoints" / "model.pt")
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_wmh2017.py",
        "--manifest",
        str(state.manifest_csv),
        "--split",
        str(state.split_csv),
        "--predictions",
        str(pred_dir),
        "--out-dir",
        str(eval_dir),
        "--run-id",
        ctx.run_id,
        "--allow-shape-only-geometry",
        "--skip-missing-predictions",
        "--model-artifact",
        str(model_dst) if model_dst.exists() else "",
        "--config-path",
        str(state.train_config),
    ]
    step = run_command(eval_cmd, cwd=ctx.repo_root)
    append_command_log(ctx.work_dir, step)
    require_ok(step)
    state.stage_status["prediction"] = "PASS"
    state.stage_status["evaluation"] = "PASS"
    eval_log = logs_dir / "eval_log.jsonl"
    eval_log.write_text(json.dumps(step, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    pred_rows = build_prediction_manifest(
        run_id=ctx.run_id,
        prediction_dir=pred_dir,
        manifest_csv=state.manifest_csv,
        split_csv=state.split_csv,
    )
    pred_manifest_path = pred_dir / "prediction_manifest.csv"
    write_prediction_manifest(pred_manifest_path, pred_rows)
    if pred_manifest_path.exists():
        state.manifest.add(
            "prediction_manifest",
            pred_manifest_path,
            producer="src/wmh2017/lineage/prediction_manifest.py",
            inputs=["predictions"],
        )
    write_prediction_label_linkage(eval_dir / "raw_prediction_label_linkage.csv", pred_rows)

    case_metrics = eval_dir / "case_metrics.csv"
    if case_metrics.exists():
        import pandas as pd

        cols = pd.read_csv(case_metrics, nrows=0).columns.tolist()
        schema_result = {
            "pass": not validate_case_metrics_columns(cols),
            "missing_columns": validate_case_metrics_columns(cols),
        }
        write_json(eval_dir / "metric_schema_validation.json", schema_result)
        state.manifest.add(
            "case_metrics",
            case_metrics,
            producer="src/wmh2017/evaluation/evaluate_predictions.py",
            inputs=["predictions", "split_manifest"],
        )
    metrics_summary = eval_dir / "metrics_summary.json"
    if metrics_summary.exists():
        state.manifest.add(
            "metrics_summary",
            metrics_summary,
            producer="src/wmh2017/evaluation/evaluate_predictions.py",
            inputs=["case_metrics"],
        )
    return state._record("evaluation", "PASS", [str(eval_dir / "case_metrics.csv")])
