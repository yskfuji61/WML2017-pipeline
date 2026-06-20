#!/usr/bin/env python3
"""Post-training workflow: threshold sweep + evaluate at best threshold (val only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.evaluate_predictions import evaluate_predictions
from wmh2017.evaluation.threshold_sweep import (
    select_best_threshold,
    sweep_thresholds,
    write_binary_predictions_at_threshold,
    write_threshold_sweep_artifacts,
)


def run_posttrain_sweep_eval(
    *,
    run_dir: str | Path,
    manifest_csv: str | Path,
    split_csv: str | Path,
    config_path: str | Path,
    assigned_split: str = "val",
    training_threshold: float = 0.5,
) -> dict:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    probs_dir = run_dir / "predictions" / "probs"
    if not probs_dir.exists() or not any(probs_dir.glob("*.npz")):
        raise FileNotFoundError(f"probability maps missing under {probs_dir}; run export_val_probabilities first")

    sweep_out = run_dir / "evaluation" / "threshold_sweep"
    summary_df, per_case_df = sweep_thresholds(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        assigned_split=assigned_split,
    )
    best = select_best_threshold(summary_df)
    payload = write_threshold_sweep_artifacts(
        out_dir=sweep_out,
        summary_df=summary_df,
        per_case_df=per_case_df,
        best=best,
        run_id=run_id,
        probs_dir=probs_dir,
        training_threshold=training_threshold,
    )

    pred_dir = run_dir / "predictions"
    write_binary_predictions_at_threshold(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        prediction_dir=pred_dir,
        threshold=float(best["threshold"]),
        assigned_split=assigned_split,
    )

    eval_dir = run_dir / "evaluation"
    ckpt = run_dir / "checkpoints" / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "model_smoke.pt"
    metrics = evaluate_predictions(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        prediction_dir=pred_dir,
        out_dir=eval_dir,
        run_id=run_id,
        assigned_split=assigned_split,
        threshold=float(best["threshold"]),
        strict_geometry=False,
        model_artifact_path=str(ckpt) if ckpt.exists() else "",
        config_path=str(config_path),
    )

    evidence_path = run_dir / "run_evidence.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["threshold_policy"] = payload["threshold_policy"]
        evidence["threshold_sweep_artifact"] = str(sweep_out / "threshold_sweep_best.json")
        evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"sweep": payload, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run val threshold sweep and evaluation for a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default="reports/dataset_manifest.csv")
    parser.add_argument("--split", default="data/splits/wmh2017_train_val_seed42.csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--assigned-split", default="val")
    parser.add_argument("--training-threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = run_posttrain_sweep_eval(
        run_dir=args.run_dir,
        manifest_csv=args.manifest,
        split_csv=args.split,
        config_path=args.config,
        assigned_split=args.assigned_split,
        training_threshold=args.training_threshold,
    )
    metrics = result["metrics"]
    best = result["sweep"]["best"]
    print(
        f"sweep_best_threshold={best['threshold']:.4f} "
        f"mean_dice={metrics['mean_dice']:.6f} "
        f"mean_lesion_recall={metrics['mean_lesion_recall']:.6f}"
    )


if __name__ == "__main__":
    main()
