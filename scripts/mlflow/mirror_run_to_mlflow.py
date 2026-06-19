#!/usr/bin/env python3
"""Mirror existing run evidence/metrics into local MLflow (thin mirror; registry remains canonical)."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

import yaml


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mirror_run_to_mlflow(
    *,
    run_dir: str | Path,
    config_path: str | Path = "configs/mlflow/local_mlflow.yaml",
) -> str:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tracking_uri = str(cfg.get("tracking_uri", "file:./mlruns"))
    if tracking_uri.startswith("http"):
        raise SystemExit("remote MLflow refused by default")

    try:
        mlflow = cast(Any, importlib.import_module("mlflow"))
    except ImportError as e:
        raise SystemExit("mlflow is required for mirror_run_to_mlflow") from e

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(cfg.get("experiment_name", "wmh2017-local-public-poc"))
    mlflow.set_experiment(experiment_name)

    evidence = _load_json(run_dir / "run_evidence.json")
    metrics = _load_json(run_dir / "evaluation" / "metrics_summary.json")
    sweep_best = _load_json(run_dir / "evaluation" / "threshold_sweep" / "threshold_sweep_best.json")

    run_id = str(evidence.get("run_id") or run_dir.name)
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tags(
            {
                "mirror_only": "true",
                "registry_is_canonical": "true",
                "project": str(cfg.get("tags", {}).get("project", "WML2017-pipeline")),
            }
        )
        params = {
            "run_id": run_id,
            "device": str(evidence.get("device", "")),
            "training_mode": str(evidence.get("training_mode", "")),
            "threshold": str(metrics.get("threshold", "")),
        }
        threshold_policy = evidence.get("threshold_policy") or sweep_best.get("threshold_policy") or {}
        if threshold_policy:
            params["sweep_best_threshold"] = str(threshold_policy.get("sweep_best_threshold", ""))
        mlflow.log_params({k: v for k, v in params.items() if v})

        metric_values = {
            "mean_dice": metrics.get("mean_dice"),
            "mean_lesion_recall": metrics.get("mean_lesion_recall"),
            "mean_lesion_f1": metrics.get("mean_lesion_f1"),
            "mean_hd95": metrics.get("mean_hd95"),
            "mean_avd": metrics.get("mean_avd"),
            "best_val_dice": evidence.get("best_val_dice"),
        }
        if sweep_best.get("best"):
            metric_values["sweep_mean_dice"] = sweep_best["best"].get("mean_dice")
            metric_values["sweep_mean_lesion_recall"] = sweep_best["best"].get("mean_lesion_recall")
        for key, value in metric_values.items():
            if value is not None:
                mlflow.log_metric(key, float(value))

        train_log = run_dir / "logs" / "train_log.jsonl"
        if train_log.exists():
            mlflow.log_text(train_log.read_text(encoding="utf-8"), artifact_file="mirror/train_log.jsonl")
        mlflow.log_dict(evidence or {"run_id": run_id}, "mirror/run_evidence.json")
        if metrics:
            mlflow.log_dict(metrics, "mirror/metrics_summary.json")
        if sweep_best:
            mlflow.log_dict(sweep_best, "mirror/threshold_sweep_best.json")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror a WMH2017 run directory into local MLflow.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/mlflow/local_mlflow.yaml")
    args = parser.parse_args()
    run_id = mirror_run_to_mlflow(run_dir=args.run_dir, config_path=args.config)
    print(f"Mirrored run_id={run_id} to local MLflow")


if __name__ == "__main__":
    main()
