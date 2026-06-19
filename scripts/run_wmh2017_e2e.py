#!/usr/bin/env python3
"""Run WMH2017 local PoC evidence chain (thin CLI wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.e2e.context import E2ERunContext
from wmh2017.e2e.runner import run_pipeline


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir or f"artifacts/runs/{args.run_id}")
    ctx = E2ERunContext(
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        files_root=args.files_root,
        seed=args.seed,
        work_dir=work_dir,
        config_path=REPO_ROOT / args.config,
        sha256sums=args.sha256sums,
        official_metrics=args.official_metrics,
        skip_train=args.skip_train,
        no_inspect_images=args.no_inspect_images,
        allow_dirty_git=args.allow_dirty_git,
    )
    result = run_pipeline(ctx)
    print(f"Wrote run directory: {result.work_dir}")
    print(f"Wrote artifact manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
