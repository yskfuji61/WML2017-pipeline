"""v4 run evidence materialization and FAILED_RUN recording."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wmh2017.lineage.hashes import sha256_path
from wmh2017.lineage.runtime_fingerprint import git_commit_or_unknown, git_dirty, package_versions


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def _csv_to_json_manifest(csv_path: Path, json_path: Path) -> bool:
    if not csv_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv_path)
    payload = {
        "format": "prediction_manifest",
        "rows": df.to_dict(orient="records"),
        "source_csv_sha256": sha256_path(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def build_run_context_v4(run_dir: Path, *, run_id: str, seed: int, status: str) -> dict[str, Any]:
    ctx_path = run_dir / "run_context.json"
    base: dict[str, Any] = {}
    if ctx_path.exists():
        base = json.loads(ctx_path.read_text(encoding="utf-8"))
    versions = package_versions()
    env_blob = json.dumps({"python": versions.get("python"), "platform": versions.get("platform")}, sort_keys=True)
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit_or_unknown(),
        "git_dirty": git_dirty(),
        "dataset_manifest_hash": base.get("dataset_manifest_sha256", "NOT_AVAILABLE"),
        "split_manifest_hash": base.get("split_manifest_sha256", "NOT_AVAILABLE"),
        "config_hash": base.get("config_hash", "NOT_AVAILABLE"),
        "env_hash": hashlib.sha256(env_blob.encode("utf-8")).hexdigest(),
        "seed": seed,
        "model_id": "monai_3d_unet_tiny_smoke",
        "release_state": "not_ready_for_release",
        "claim_boundary": "public_data_local_poc_only",
        "status": status,
    }


def write_evidence_summary(run_dir: Path, *, run_id: str, status: str) -> Path:
    metrics_path = run_dir / "metrics_summary.json"
    dice = "NOT_AVAILABLE"
    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        dice = summary.get("dice_mean", summary.get("metrics", {}).get("dice_local", "NOT_AVAILABLE"))
    lines = [
        f"# Evidence summary — {run_id}",
        "",
        f"- status: {status}",
        "- claim_boundary: public_data_local_poc_only",
        "- local validation only; not official benchmark, clinical, customer, production, SOTA, or READY_FOR_RELEASE",
        f"- dice_local_or_equivalent: {dice}",
        "",
    ]
    out = run_dir / "evidence_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def materialize_v4_artifacts(
    run_dir: Path,
    *,
    run_id: str,
    seed: int,
    status: str = "COMPLETED_OR_FAILED_RUN",
    repo_root: Path | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    v4_ctx = build_run_context_v4(run_dir, run_id=run_id, seed=seed, status=status)
    (run_dir / "run_context.json").write_text(json.dumps(v4_ctx, indent=2, ensure_ascii=False), encoding="utf-8")

    _copy_if_exists(run_dir / "runtime_fingerprint.json", run_dir / "runtime_fingerprint.json")
    if not (run_dir / "runtime_fingerprint.json").exists():
        from wmh2017.lineage.runtime_fingerprint import write_runtime_fingerprint

        root = repo_root or Path(__file__).resolve().parents[2]
        write_runtime_fingerprint(run_dir / "runtime_fingerprint.json", repo_root=root)

    _copy_if_exists(run_dir / "dataset" / "dataset_manifest.json", run_dir / "dataset_manifest.json")
    split_src = run_dir / "splits" / "split_manifest.json"
    if split_src.exists() and split_src.suffix == ".json":
        try:
            json.loads(split_src.read_text(encoding="utf-8"))
            _copy_if_exists(split_src, run_dir / "split_manifest.json")
        except json.JSONDecodeError:
            pass
    split_csv = next(run_dir.glob("splits/wmh2017_train_val_seed*.csv"), None)
    if split_csv and not (run_dir / "split_manifest.json").exists():
        import pandas as pd

        df = pd.read_csv(split_csv)
        payload = {
            "split_id": f"WMH2017-TRAIN-VAL-SEED{seed}",
            "seed": seed,
            "train_case_ids": df[df["assigned_split"] == "train"]["case_id"].astype(str).tolist()
            if "assigned_split" in df.columns
            else [],
            "val_case_ids": df[df["assigned_split"] == "val"]["case_id"].astype(str).tolist()
            if "assigned_split" in df.columns
            else [],
        }
        payload["split_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        (run_dir / "split_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _copy_if_exists(run_dir / "configs" / "train_config.materialized.yaml", run_dir / "train_config.materialized.yaml")
    pred_csv = run_dir / "predictions" / "prediction_manifest.csv"
    if not _csv_to_json_manifest(pred_csv, run_dir / "prediction_manifest.json"):
        if pred_csv.exists():
            _copy_if_exists(pred_csv, run_dir / "prediction_manifest.csv")

    _copy_if_exists(run_dir / "evaluation" / "metrics_summary.json", run_dir / "metrics_summary.json")
    _copy_if_exists(run_dir / "evaluation" / "case_metrics.csv", run_dir / "case_metrics.csv")
    write_evidence_summary(run_dir, run_id=run_id, status=status)

    return {
        k: str(run_dir / k)
        for k in (
            "run_context.json",
            "runtime_fingerprint.json",
            "dataset_manifest.json",
            "split_manifest.json",
            "train_config.materialized.yaml",
            "prediction_manifest.json",
            "metrics_summary.json",
            "case_metrics.csv",
            "evidence_summary.md",
        )
    }


def record_failed_run(run_dir: Path, *, run_id: str, seed: int, reason: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "run_id": run_id,
        "status": "FAILED_RUN",
        "reason": reason,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "failed_run.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    materialize_v4_artifacts(run_dir, run_id=run_id, seed=seed, status="FAILED_RUN")


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
