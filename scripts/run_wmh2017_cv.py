#!/usr/bin/env python3
"""Run a WMH2017 k-fold cross-validation sequentially and aggregate results.

For each fold this runs the full e2e pipeline (train -> predict -> evaluate) using
a per-fold config whose data.split_manifest points to a committed CV fold, then runs
the validation-only threshold sweep + evaluation, and finally aggregates per-fold
validation metrics into mean +/- std.

Local cross-validation only; never uses the test split; no SOTA/clinical claim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.evaluation.cv_aggregate import collect_fold_summaries, write_cv_summary


def _run(cmd: list[str]) -> None:
    print(f"[cv] $ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"command failed (rc={completed.returncode}): {' '.join(cmd)}")


def _fold_config(config_dir: Path, prefix: str, fold: int) -> Path:
    path = config_dir / f"{prefix}{fold}.yaml"
    if not path.exists():
        raise SystemExit(f"fold config not found: {path}")
    return path


def _run_dir_for(config_path: Path) -> str:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(cfg["run"]["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WMH2017 k-fold CV and aggregate (local validation only).")
    parser.add_argument("--files-root", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--config-dir", default="configs/experiments/cv")
    parser.add_argument("--config-prefix", default="exp_a2cv_cosine_fold")
    parser.add_argument("--fold-split-prefix", default="fold")
    parser.add_argument("--cv-id", default="wmh2017_a2cv_cosine_seed42")
    parser.add_argument("--out", default="reports/cv/cv_summary_a2cv_cosine_seed42.json")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--allow-dirty-git", action="store_true")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only run sweep+aggregate on existing runs.")
    args = parser.parse_args()

    config_dir = REPO_ROOT / args.config_dir
    run_dirs: list[Path] = []
    started = time.time()
    for fold in range(args.k):
        cfg_path = _fold_config(config_dir, args.config_prefix, fold)
        run_id = _run_dir_for(cfg_path)
        run_dir = REPO_ROOT / "artifacts" / "runs" / run_id
        run_dirs.append(run_dir)
        if fold < args.start_fold:
            print(f"[cv] skipping fold {fold} (start-fold={args.start_fold})", flush=True)
            continue

        print(f"\n[cv] ===== fold {fold} run_id={run_id} =====", flush=True)
        if not args.skip_train:
            e2e_cmd = [
                sys.executable,
                "scripts/run_wmh2017_e2e.py",
                "--files-root",
                args.files_root,
                "--run-id",
                run_id,
                "--config",
                str(cfg_path.relative_to(REPO_ROOT)),
            ]
            if args.max_epochs is not None:
                e2e_cmd += ["--max-epochs", str(args.max_epochs)]
            if args.allow_dirty_git:
                e2e_cmd.append("--allow-dirty-git")
            if args.overwrite_run:
                e2e_cmd.append("--overwrite-run")
            _run(e2e_cmd)

        # Validation-only threshold sweep + evaluation at best threshold.
        fold_split = run_dir / "splits" / f"{args.fold_split_prefix}{fold}.csv"
        dataset_manifest = run_dir / "dataset" / "dataset_manifest.csv"
        sweep_cmd = [
            sys.executable,
            "scripts/run_posttrain_sweep_eval.py",
            "--run-dir",
            str(run_dir.relative_to(REPO_ROOT)),
            "--config",
            str(cfg_path.relative_to(REPO_ROOT)),
            "--manifest",
            str(dataset_manifest.relative_to(REPO_ROOT)),
            "--split",
            str(fold_split.relative_to(REPO_ROOT)),
            "--assigned-split",
            "val",
        ]
        _run(sweep_cmd)

    summaries = collect_fold_summaries(run_dirs)
    payload = write_cv_summary(REPO_ROOT / args.out, summaries, cv_id=args.cv_id)
    elapsed = time.time() - started
    print(f"\n[cv] wrote {args.out} in {elapsed / 3600.0:.2f}h", flush=True)
    md = payload["metrics"]["mean_dice"]
    rc = payload["metrics"]["mean_lesion_recall"]
    print(
        f"[cv] CV mean_dice={md['mean']:.4f}+/-{md['std']:.4f} "
        f"mean_lesion_recall={rc['mean']:.4f}+/-{rc['std']:.4f} (n_folds={payload['n_folds']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
