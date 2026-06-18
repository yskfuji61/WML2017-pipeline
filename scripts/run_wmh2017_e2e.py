#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.evaluation.metric_schema import validate_case_metrics_columns
from wmh2017.lineage.artifact_manifest import ArtifactManifest
from wmh2017.lineage.hashes import sha256_path, write_hash_sidecar, write_json
from wmh2017.lineage.lineage_graph import (
    artifact_hashes_from_manifest,
    build_lineage_graph,
    write_lineage_graph,
)
from wmh2017.lineage.prediction_manifest import (
    build_prediction_manifest,
    write_prediction_label_linkage,
    write_prediction_manifest,
)
from wmh2017.lineage.run_context import append_command_log, init_run_directory, update_run_context
from wmh2017.lineage.runtime_fingerprint import git_dirty, write_runtime_fingerprint
from wmh2017.observability.run_observability import build_run_observability, write_run_observability


def _run(cmd: Sequence[str], *, cwd: Path) -> dict:
    completed = subprocess.run(list(cmd), cwd=str(cwd), text=True, capture_output=True, check=False)
    return {
        "cmd": list(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _require_ok(step: dict) -> None:
    if int(step["returncode"]) != 0:
        raise SystemExit(
            "command failed:\n"
            + " ".join(step["cmd"])
            + "\nSTDOUT:\n"
            + step["stdout"]
            + "\nSTDERR:\n"
            + step["stderr"]
        )


def _copy_to_nested(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    write_hash_sidecar(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WMH2017 local PoC evidence chain.")
    parser.add_argument("--files-root", required=True)
    parser.add_argument("--run-id", default="wmh2017_preview_20260618_unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--config", default="configs/wmh2017_monai_smoke.yaml")
    parser.add_argument("--sha256sums", default="evidence/wmh2017_download_2026-06-16/SHA256SUMS.txt")
    parser.add_argument("--official-metrics", default="")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-inspect-images", action="store_true")
    parser.add_argument("--allow-dirty-git", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir or f"artifacts/runs/{args.run_id}")
    init_run_directory(work_dir, run_id=args.run_id, wmh2017_root=args.files_root, seed=args.seed)
    write_runtime_fingerprint(work_dir / "runtime_fingerprint.json", repo_root=REPO_ROOT)

    if git_dirty() and not args.allow_dirty_git:
        raise SystemExit("git working tree is dirty; commit changes or pass --allow-dirty-git")

    dataset_dir = work_dir / "dataset"
    label_dir = work_dir / "label_audit"
    splits_dir = work_dir / "splits"
    configs_dir = work_dir / "configs"
    logs_dir = work_dir / "logs"
    ckpt_dir = work_dir / "checkpoints"
    pred_dir = work_dir / "predictions"
    eval_dir = work_dir / "evaluation"
    manifest = ArtifactManifest(args.run_id)

    manifest_csv = dataset_dir / "dataset_manifest.csv"
    manifest_json = dataset_dir / "dataset_manifest.json"
    manifest_summary = dataset_dir / "dataset_manifest.summary.json"
    dataset_cmd = [
        sys.executable,
        "scripts/audit_wmh2017_dataset.py",
        "--root",
        args.files_root,
        "--out",
        str(manifest_csv),
        "--summary-out",
        str(manifest_summary),
        "--strict-counts",
        "--sha256sums",
        args.sha256sums,
    ]
    if not args.no_inspect_images:
        dataset_cmd.extend(["--inspect-images", "--fail-on-metadata-error"])
    step = _run(dataset_cmd, cwd=REPO_ROOT)
    append_command_log(work_dir, step)
    _require_ok(step)
    _copy_to_nested(manifest_csv, manifest_json)
    manifest.add("dataset_manifest", manifest_json, producer="scripts/audit_wmh2017_dataset.py")

    label_csv = label_dir / "label_audit.csv"
    label_json = label_dir / "label_audit.json"
    label_summary = label_dir / "label_audit.summary.json"
    label_cmd = [
        sys.executable,
        "scripts/audit_wmh2017_labels.py",
        "--manifest",
        str(manifest_csv),
        "--out",
        str(label_csv),
        "--summary-out",
        str(label_summary),
        "--split",
        "training",
        "--include-geometry",
        "--fail-on-error",
    ]
    step = _run(label_cmd, cwd=REPO_ROOT)
    append_command_log(work_dir, step)
    _require_ok(step)
    _copy_to_nested(label_csv, label_json)
    manifest.add("label_audit", label_json, producer="scripts/audit_wmh2017_labels.py", inputs=["dataset_manifest"])

    split_cmd = [
        sys.executable,
        "scripts/make_wmh2017_splits.py",
        "--manifest",
        str(manifest_csv),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(splits_dir),
    ]
    step = _run(split_cmd, cwd=REPO_ROOT)
    append_command_log(work_dir, step)
    _require_ok(step)
    split_csv = splits_dir / f"wmh2017_train_val_seed{args.seed}.csv"
    split_manifest = splits_dir / "split_manifest.json"
    split_manifest.write_text(split_csv.read_text(encoding="utf-8"), encoding="utf-8")
    write_hash_sidecar(split_manifest)
    manifest.add("split_manifest", split_manifest, producer="scripts/make_wmh2017_splits.py", inputs=["dataset_manifest"])

    update_run_context(
        work_dir,
        dataset_manifest_hash=sha256_path(manifest_json),
        split_manifest_hash=sha256_path(split_manifest),
    )

    stage_status = {"dataset_audit": "PASS", "label_audit": "PASS"}
    stage_status["split"] = "PASS"

    if not args.skip_train:
        import yaml

        train_config = configs_dir / "train_config.materialized.yaml"
        eval_config = configs_dir / "eval_config.materialized.yaml"
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        cfg.setdefault("run", {})
        cfg.setdefault("data", {})
        cfg["run"].update(
            {
                "run_id": args.run_id,
                "seed": args.seed,
                "output_dir": work_dir.as_posix(),
                "run_manifest": "registry/runs/run_manifest.csv",
            }
        )
        cfg["data"].update(
            {
                "dataset_manifest": manifest_csv.as_posix(),
                "split_manifest": split_csv.as_posix(),
            }
        )
        train_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_hash_sidecar(train_config)
        eval_cfg = {
            "run_id": args.run_id,
            "threshold": 0.5,
            "assigned_split": "val",
            "manifest_csv": manifest_csv.as_posix(),
            "split_csv": split_csv.as_posix(),
            "prediction_dir": pred_dir.as_posix(),
        }
        eval_config.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")
        write_hash_sidecar(eval_config)
        config_snapshot = configs_dir / "config_snapshot.sha256"
        combined = sha256_path(train_config) + sha256_path(eval_config)
        config_snapshot.write_text(combined + "\n", encoding="utf-8")
        manifest.add(
            "train_config",
            train_config,
            producer="scripts/run_wmh2017_e2e.py",
            inputs=["dataset_manifest", "split_manifest"],
        )
        update_run_context(
            work_dir,
            config_hash=sha256_path(train_config),
            config_snapshot_sha256=config_snapshot.read_text(encoding="utf-8").strip(),
        )

        train_cmd = [sys.executable, "scripts/train_wmh2017.py", "--config", str(train_config)]
        step = _run(train_cmd, cwd=REPO_ROOT)
        append_command_log(work_dir, step)
        _require_ok(step)
        stage_status["training"] = "PASS"

        model_src = work_dir / "checkpoints" / "model_smoke.pt"
        model_dst = ckpt_dir / "model.pt"
        if model_src.exists():
            _copy_to_nested(model_src, model_dst)
            manifest.add(
                "model",
                model_dst,
                producer="src/wmh2017/training/train_monai.py",
                inputs=["split_manifest", "train_config"],
            )
            model_card = ckpt_dir / "model_card_fragment.json"
            write_json(
                model_card,
                {
                    "run_id": args.run_id,
                    "model_artifact_sha256": sha256_path(model_dst),
                    "claim_boundary": "smoke training only; no performance claim",
                },
            )

        train_log_src = work_dir / "logs" / "train_log.jsonl"
        if train_log_src.exists():
            _copy_to_nested(train_log_src, logs_dir / "train_log.jsonl")

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_wmh2017.py",
            "--manifest",
            str(manifest_csv),
            "--split",
            str(split_csv),
            "--predictions",
            str(pred_dir),
            "--out-dir",
            str(eval_dir),
            "--run-id",
            args.run_id,
            "--allow-shape-only-geometry",
            "--model-artifact",
            str(model_dst) if model_dst.exists() else "",
            "--config-path",
            str(train_config),
        ]
        step = _run(eval_cmd, cwd=REPO_ROOT)
        append_command_log(work_dir, step)
        _require_ok(step)
        stage_status["prediction"] = "PASS"
        stage_status["evaluation"] = "PASS"

        eval_log = logs_dir / "eval_log.jsonl"
        eval_log.write_text(json.dumps(step, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

        pred_rows = build_prediction_manifest(
            run_id=args.run_id,
            prediction_dir=pred_dir,
            manifest_csv=manifest_csv,
            split_csv=split_csv,
        )
        pred_manifest_path = pred_dir / "prediction_manifest.csv"
        write_prediction_manifest(pred_manifest_path, pred_rows)
        if pred_manifest_path.exists():
            manifest.add(
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
            manifest.add(
                "case_metrics",
                case_metrics,
                producer="src/wmh2017/evaluation/evaluate_predictions.py",
                inputs=["predictions", "split_manifest"],
            )
        metrics_summary = eval_dir / "metrics_summary.json"
        if metrics_summary.exists():
            manifest.add(
                "metrics_summary",
                metrics_summary,
                producer="src/wmh2017/evaluation/evaluate_predictions.py",
                inputs=["case_metrics"],
            )

    manifest_path = work_dir / "artifact_manifest.json"
    manifest.write(manifest_path)
    write_hash_sidecar(manifest_path)

    graph = build_lineage_graph(
        run_id=args.run_id,
        artifact_hashes=artifact_hashes_from_manifest(manifest_path),
        package_version="0.2.3",
    )
    write_lineage_graph(work_dir / "lineage" / "lineage_graph.json", graph)

    stage_status["split_generation"] = stage_status.pop("split", "PASS")
    stage_status["prediction_export"] = stage_status.pop("prediction", "SKIP")
    stage_status["lineage_verification"] = "PENDING"
    stage_status["binder_verification"] = "PENDING"

    observability = build_run_observability(
        run_id=args.run_id,
        release_state="PREVIEW_CANDIDATE",
        dataset_summary={"status": stage_status.get("dataset_audit", "SKIP")},
        training_summary={"status": stage_status.get("training", "SKIP")},
        inference_summary={"status": stage_status.get("prediction_export", "SKIP")},
        evaluation_summary={"status": stage_status.get("evaluation", "SKIP")},
    )
    observability["status"] = "PASS"
    observability["stage_status"] = stage_status
    write_run_observability(work_dir / "observability/offline_run_summary.json", observability)

    print(f"Wrote run directory: {work_dir}")
    print(f"Wrote artifact manifest: {manifest_path}")


if __name__ == "__main__":
    main()
