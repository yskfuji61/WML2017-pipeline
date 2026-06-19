#!/usr/bin/env python3
"""Generate one-case overlay locally and write figure manifest only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-case overlay and figure manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", default="reports/figures/overlays")
    parser.add_argument("--manifest-out", default="artifacts/manifests/figure_manifest.json")
    parser.add_argument("--commit-png", action="store_true", help="Not recommended; default is manifest only")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{args.case_id}_flair_label1_overlay.png"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "visualize_wmh_case.py"),
        "--manifest",
        args.manifest,
        "--case-id",
        args.case_id,
        "--out",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "GENERATED" if result.returncode == 0 and png_path.exists() else "FAILED"

    figure_manifest = {
        "case_id": args.case_id,
        "run_id": "PENDING_CONFIRMATION",
        "audit_id": f"one_case_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "source_image_hash": "PENDING_CONFIRMATION",
        "label_hash": "PENDING_CONFIRMATION",
        "generated_file_hash": sha256_file(png_path) if png_path.exists() else "NOT_AVAILABLE",
        "public_data_only": True,
        "local_path_redacted": "REDACTED_OR_LOCAL_ONLY",
        "png_committed": bool(args.commit_png),
        "status": status,
        "title_must_not_include_absolute_path": True,
    }
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    print(f"Wrote figure manifest {manifest_out}")
    if status == "FAILED":
        raise SystemExit(result.stderr or result.stdout or "overlay generation failed")


if __name__ == "__main__":
    main()
