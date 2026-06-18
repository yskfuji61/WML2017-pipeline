#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EXCLUDE_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

REQUIRED_FILES = [
    "README.md",
    "README_ja.md",
    "README_CURSOR_START.md",
    "pyproject.toml",
    "requirements-lock.txt",
    "configs/wmh2017_monai_smoke.yaml",
    "scripts/audit_wmh2017_dataset.py",
    "scripts/audit_wmh2017_labels.py",
    "scripts/make_wmh2017_splits.py",
    "scripts/visualize_wmh_case.py",
    "scripts/train_wmh2017.py",
    "scripts/evaluate_wmh2017.py",
    "scripts/verify_wmh2017_download_evidence.py",
    "scripts/run_wmh2017_e2e.py",
    "scripts/compare_official_evaluator_parity.py",
    "src/wmh2017/training/train_monai.py",
    "src/wmh2017/evaluation/evaluate_predictions.py",
    "src/wmh2017/evaluation/official_parity.py",
    "docs/engineering_validation_plan.md",
    "docs/release_decision_record.md",
    "docs/release_state_crosswalk.md",
    "docs/final_evidence_binder_index.md",
    "registry/finding_register_wmh2017.csv",
    "registry/review_approval_register_wmh2017.csv",
    "registry/decision_record_register_wmh2017.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_manifest_files(root: Path, excluded_rel: str | None = None) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if excluded_rel is not None and rel == excluded_rel:
            continue
        if any(part in EXCLUDE_DIR_PARTS for part in p.relative_to(root).parts):
            continue
        files.append(p)
    return sorted(files, key=lambda x: x.relative_to(root).as_posix())


def default_status(rel: str) -> str:
    if rel.startswith("artifacts/") or rel.startswith("data/splits/") or rel.startswith("reports/metrics/") or rel.startswith("reports/overlays/") or rel.startswith("reports/runs/"):
        return "placeholder_or_generated_area"
    if rel.startswith("core/pipeline/"):
        return "inherited_reference_not_primary_smoke_path"
    if rel.startswith("docs/future_sota/"):
        return "future_plan_blocked_until_exp000"
    return "current_scaffold_file"


def default_owner(rel: str) -> str:
    if rel.startswith("registry/source") or rel.startswith("docs/source"):
        return "source_license_owner"
    if rel.startswith("docs/security") or rel.startswith(".secrets"):
        return "security_privacy_owner"
    if rel.startswith("src/") or rel.startswith("scripts/") or rel.startswith("configs/") or rel.startswith("tests/"):
        return "implementation_lead"
    if rel.startswith("docs/") or rel.startswith("registry/") or rel.startswith("reports/"):
        return "evidence_owner"
    return "documentation_owner"



def relative_to_root(path: Path, root: Path, option_name: str) -> str:
    """Return a repo-relative path or fail with an actionable CLI error."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as e:
        raise SystemExit(
            f"{option_name} must resolve inside --repo-root for a stable package manifest: "
            f"{resolved} is outside {root}"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate expected repo files and write a full package manifest.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="reports/full_package_manifest.json")
    parser.add_argument("--package-id", default="WMH2017-LOCAL-POC-SCAFFOLD-0.2.2")
    parser.add_argument("--package-version", default="0.2.2")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    missing = [p for p in REQUIRED_FILES if not (root / p).exists()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    rows = []
    output_rel = relative_to_root(root / args.out, root, "--out")
    for p in iter_manifest_files(root, excluded_rel=output_rel):
        rel = p.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "owner": default_owner(rel),
                "status": default_status(rel),
                "dlp_class": "public_or_internal_scaffold_no_raw_medical_images",
                "release_boundary": "file identity only; not training/evaluation/approval evidence",
            }
        )

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_id": args.package_id,
        "package_version": args.package_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_structural_manifest_created",
        "file_count": len(rows),
        "claim_boundary": "Every included file is identified by path, bytes, and sha256. This is not proof of real WMH2017 training/evaluation, source review, Preview, or release approval.",
        "required_files_checked": REQUIRED_FILES,
        "manifest_file_excluded_from_own_file_list_for_stable_hash": output_rel,
        "files": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} files")


if __name__ == "__main__":
    main()
