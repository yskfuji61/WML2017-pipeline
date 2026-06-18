#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.lineage.artifact_manifest import ArtifactManifest
from wmh2017.lineage.hashes import write_hash_sidecar
from wmh2017.lineage.run_context import append_command_log, init_run_directory
from wmh2017.lineage.runtime_fingerprint import write_runtime_fingerprint


def _run(cmd: Sequence[str], *, cwd: Path) -> dict:
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
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


def _copy_with_hash(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    write_hash_sidecar(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the WMH2017 local PoC evidence chain: dataset manifest, label audit, split, "
            "MONAI training/prediction, local validation metrics, optional official parity comparison."
        )
    )
    parser.add_argument("--files-root", required=True, help="Raw Dataverse files/ root or parent containing files/.")
    parser.add_argument("--run-id", default="wmh2017_local_e2e_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--config", default="configs/wmh2017_monai_smoke.yaml")
    parser.add_argument("--sha256sums", default="evidence/wmh2017_download_2026-06-16/SHA256SUMS.txt")
    parser.add_argument("--official-metrics", default="", help="Optional official evaluator case-level export.")
    parser.add_argument("--skip-train", action="store_true", help="Generate audits and split only.")
    parser.add_argument("--no-inspect-images", action="store_true", help="Skip NIfTI geometry inspection during dataset audit.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir or f"artifacts/runs/{args.run_id}")
    init_run_directory(work_dir, run_id=args.run_id, wmh2017_root=args.files_root, seed=args.seed)
    write_runtime_fingerprint(work_dir / "runtime_fingerprint.json", repo_root=REPO_ROOT)

    manifest_json = work_dir / "dataset_manifest.json"
    manifest_csv = work_dir / "dataset_manifest.csv"
    manifest_summary = work_dir / "dataset_manifest.summary.json"
    label_audit_json = work_dir / "label_audit.json"
    label_audit_csv = work_dir / "label_value_audit.csv"
    label_summary = work_dir / "label_value_audit.summary.json"
    split_manifest = work_dir / "split_manifest.json"
    split_dir = work_dir / "splits"
    eval_dir = work_dir / "evaluation/local"
    parity_dir = work_dir / "evaluation/official_parity"
    manifest = ArtifactManifest(args.run_id)

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
    _copy_with_hash(manifest_csv, manifest_json)
    manifest.add("dataset_manifest", manifest_json, producer="scripts/audit_wmh2017_dataset.py")

    label_cmd = [
        sys.executable,
        "scripts/audit_wmh2017_labels.py",
        "--manifest",
        str(manifest_csv),
        "--out",
        str(label_audit_csv),
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
    _copy_with_hash(label_audit_csv, label_audit_json)
    manifest.add("label_audit", label_audit_json, producer="scripts/audit_wmh2017_labels.py", inputs=["dataset_manifest"])

    split_cmd = [
        sys.executable,
        "scripts/make_wmh2017_splits.py",
        "--manifest",
        str(manifest_csv),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(split_dir),
    ]
    step = _run(split_cmd, cwd=REPO_ROOT)
    append_command_log(work_dir, step)
    _require_ok(step)
    split_csv = split_dir / f"wmh2017_train_val_seed{args.seed}.csv"
    split_manifest.write_text(split_csv.read_text(encoding="utf-8"), encoding="utf-8")
    write_hash_sidecar(split_manifest)
    manifest.add("split_manifest", split_manifest, producer="scripts/make_wmh2017_splits.py", inputs=["dataset_manifest"])

    if not args.skip_train:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required to materialize the training config.") from exc

        materialized_config = work_dir / "train_config.materialized.yaml"
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        cfg.setdefault("run", {})
        cfg.setdefault("data", {})
        cfg["run"].update({
            "run_id": args.run_id,
            "seed": args.seed,
            "output_dir": work_dir.as_posix(),
            "run_manifest": "registry/runs/run_manifest.csv",
        })
        cfg["data"].update({
            "dataset_manifest": manifest_csv.as_posix(),
            "split_manifest": split_csv.as_posix(),
        })
        materialized_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_hash_sidecar(materialized_config)
        manifest.add("train_config", materialized_config, producer="scripts/run_wmh2017_e2e.py", inputs=["dataset_manifest", "split_manifest"])

        train_cmd = [sys.executable, "scripts/train_wmh2017.py", "--config", str(materialized_config)]
        step = _run(train_cmd, cwd=REPO_ROOT)
        append_command_log(work_dir, step)
        _require_ok(step)

        pred_dir = work_dir / "predictions"
        model_path = work_dir / "model" / "model_smoke.pt"
        ckpt_src = work_dir / "checkpoints" / "model_smoke.pt"
        if ckpt_src.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt_src, model_path)
            write_hash_sidecar(model_path)
            manifest.add("model", model_path, producer="src/wmh2017/training/train_monai.py", inputs=["split_manifest", "train_config"])

        train_log_src = work_dir / "logs" / "train_log.jsonl"
        if train_log_src.exists():
            shutil.copy2(train_log_src, work_dir / "training_log.jsonl")

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
            str(model_path) if model_path.exists() else "",
            "--config-path",
            str(materialized_config),
        ]
        step = _run(eval_cmd, cwd=REPO_ROOT)
        append_command_log(work_dir, step)
        _require_ok(step)
        case_metrics = eval_dir / "case_metrics.csv"
        metrics_summary = eval_dir / "metrics_summary.json"
        if case_metrics.exists():
            manifest.add("case_metrics", case_metrics, producer="src/wmh2017/evaluation/evaluate_predictions.py", inputs=["predictions", "split_manifest"])
        if metrics_summary.exists():
            manifest.add("metrics_summary", metrics_summary, producer="src/wmh2017/evaluation/evaluate_predictions.py", inputs=["case_metrics"])

        if args.official_metrics:
            parity_cmd = [
                sys.executable,
                "scripts/compare_official_evaluator_parity.py",
                "--local",
                str(case_metrics),
                "--official",
                args.official_metrics,
                "--out-dir",
                str(parity_dir),
            ]
            step = _run(parity_cmd, cwd=REPO_ROOT)
            append_command_log(work_dir, step)
            _require_ok(step)

    manifest.write(work_dir / "artifact_manifest.json")
    print(f"Wrote run directory: {work_dir}")
    print(f"Wrote artifact manifest: {work_dir / 'artifact_manifest.json'}")


if __name__ == "__main__":
    main()
