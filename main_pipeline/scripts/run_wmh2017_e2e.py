#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--work-dir", default="artifacts/runs/wmh2017_local_e2e_seed42")
    parser.add_argument("--config", default="configs/train/smoke_monai_unet3d.yaml")
    parser.add_argument("--sha256sums", default="evidence/wmh2017_download_2026-06-16/SHA256SUMS.txt")
    parser.add_argument("--official-metrics", default="", help="Optional official evaluator case-level export.")
    parser.add_argument("--skip-train", action="store_true", help="Generate audits and split only.")
    parser.add_argument("--no-inspect-images", action="store_true", help="Skip NIfTI geometry inspection during dataset audit.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = work_dir / "dataset_manifest.csv"
    manifest_summary = work_dir / "dataset_manifest.summary.json"
    label_audit_csv = work_dir / "label_value_audit.csv"
    label_summary = work_dir / "label_value_audit.summary.json"
    split_dir = work_dir / "splits"
    eval_dir = work_dir / "metrics"
    parity_dir = work_dir / "official_parity"
    command_log = work_dir / "e2e_command_log.json"

    steps: list[dict] = []

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
        dataset_cmd.append("--inspect-images")
        dataset_cmd.append("--fail-on-metadata-error")
    step = _run(dataset_cmd, cwd=REPO_ROOT)
    steps.append(step)
    _require_ok(step)

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
    steps.append(step)
    _require_ok(step)

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
    steps.append(step)
    _require_ok(step)

    if not args.skip_train:
        # Keep the existing YAML as source of defaults, but produce a run-specific
        # materialized config so evidence paths are explicit and reproducible.
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required to materialize the training config.") from exc

        config_path = work_dir / "materialized_train_config.yaml"
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        cfg.setdefault("run", {})
        cfg.setdefault("data", {})
        cfg["run"].update({
            "run_id": args.run_id,
            "seed": args.seed,
            "output_dir": f"{work_dir.as_posix()}/train",
            "run_manifest": f"{work_dir.as_posix()}/run_manifest.csv",
        })
        cfg["data"].update({
            "dataset_manifest": manifest_csv.as_posix(),
            "split_manifest": (split_dir / f"wmh2017_train_val_seed{args.seed}.csv").as_posix(),
        })
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        train_cmd = [sys.executable, "scripts/train_wmh2017.py", "--config", str(config_path)]
        step = _run(train_cmd, cwd=REPO_ROOT)
        steps.append(step)
        _require_ok(step)

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_wmh2017.py",
            "--manifest",
            str(manifest_csv),
            "--split",
            str(split_dir / f"wmh2017_train_val_seed{args.seed}.csv"),
            "--predictions",
            str(work_dir / "train" / "predictions"),
            "--out-dir",
            str(eval_dir),
            "--run-id",
            args.run_id,
        ]
        step = _run(eval_cmd, cwd=REPO_ROOT)
        steps.append(step)
        _require_ok(step)

        if args.official_metrics:
            parity_cmd = [
                sys.executable,
                "scripts/compare_official_evaluator_parity.py",
                "--local",
                str(eval_dir / "case_metrics.csv"),
                "--official",
                args.official_metrics,
                "--out-dir",
                str(parity_dir),
            ]
            step = _run(parity_cmd, cwd=REPO_ROOT)
            steps.append(step)
            _require_ok(step)

    command_log.write_text(json.dumps(steps, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote E2E command log: {command_log}")
    print(f"Wrote dataset manifest: {manifest_csv}")
    print(f"Wrote label audit: {label_audit_csv}")
    print(f"Wrote splits: {split_dir}")
    if not args.skip_train:
        print(f"Wrote local metrics: {eval_dir}")


if __name__ == "__main__":
    main()
