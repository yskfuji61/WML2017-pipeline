#!/usr/bin/env python3
"""Sync v4 redacted JSON manifests from canonical CSV sources.

Canonical sources:
- reports/dataset_manifest.csv (local, gitignored; optional)
- data/splits/wmh2017_train_val_seed42.csv (tracked)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_manifest_json(split_csv: Path, *, source_csv_rel: str) -> dict:
    rows = _read_csv_rows(split_csv)
    train_count = sum(1 for r in rows if str(r.get("assigned_split", "")).lower() == "train")
    val_count = sum(1 for r in rows if str(r.get("assigned_split", "")).lower() == "val")
    split_id = rows[0].get("split_id", "UNKNOWN") if rows else "UNKNOWN"
    seed = int(rows[0].get("seed", "0") or 0) if rows else 0
    return {
        "split_id": split_id,
        "seed": seed,
        "status": "OK",
        "source_csv": source_csv_rel,
        "train_count": train_count,
        "val_count": val_count,
        "policy": "training_only_no_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_hash": _sha256_path(split_csv),
    }


def _parse_shape(value: str) -> list[int]:
    text = (value or "").strip()
    if not text:
        return [0, 0, 0]
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    try:
        vals = [int(float(p)) for p in parts[:3]]
        return vals + [0] * max(0, 3 - len(vals))
    except ValueError:
        return [0, 0, 0]


def _parse_spacing(value: str) -> list[float]:
    text = (value or "").strip()
    if not text:
        return [0.0, 0.0, 0.0]
    parts = [p.strip() for p in text.replace("x", ",").replace("[", "").replace("]", "").split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts[:3]]
        return vals + [0.0] * max(0, 3 - len(vals))
    except ValueError:
        return [0.0, 0.0, 0.0]


def _dataset_manifest_json(dataset_csv: Path, *, source_csv_rel: str) -> dict:
    rows = _read_csv_rows(dataset_csv)
    cases: list[dict] = []
    for row in rows:
        t1_path = row.get("t1_pre_path") or row.get("t1_path") or ""
        cases.append(
            {
                "case_id": str(row.get("case_id", "")),
                "site": str(row.get("site", "UNSPECIFIED")),
                "challenge_split": str(row.get("challenge_split", "")),
                "modalities": {
                    "flair": {
                        "path": "REDACTED_OR_LOCAL_ONLY",
                        "sha256": str(row.get("flair_sha256") or row.get("flair_pre_sha256") or ""),
                        "shape": _parse_shape(row.get("flair_shape") or row.get("flair_pre_shape") or ""),
                        "spacing": _parse_spacing(row.get("flair_spacing") or row.get("flair_pre_spacing") or ""),
                    },
                    "t1": {
                        "path": "REDACTED_OR_LOCAL_ONLY" if t1_path else "REDACTED_OR_LOCAL_ONLY_OR_NULL",
                        "sha256": str(row.get("t1_sha256") or row.get("t1_pre_sha256") or "") or None,
                    },
                },
                "label": {
                    "path": "REDACTED_OR_LOCAL_ONLY",
                    "sha256": str(row.get("wmh_sha256") or row.get("mask_sha256") or ""),
                    "label_policy": "label==1 foreground; label==2 ignored",
                },
            }
        )
    payload = {
        "dataset_id": "wmh2017_public_poc",
        "status": "OK",
        "root": "REDACTED_OR_LOCAL_ONLY",
        "source_csv": source_csv_rel,
        "case_count": len(cases),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dlp_class": "PUBLIC_CHALLENGE_DATA",
        "cases": cases,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _missing_dataset_json(*, split_csv_rel: str, split_csv: Path) -> dict:
    split_hash = _sha256_path(split_csv) if split_csv.exists() else "MISSING_SPLIT_CSV"
    return {
        "dataset_id": "wmh2017_public_poc",
        "status": "LOCAL_CSV_REQUIRED",
        "root": "REDACTED_OR_LOCAL_ONLY",
        "source_csv": "reports/dataset_manifest.csv",
        "cases": [],
        "case_count": 0,
        "manifest_hash": hashlib.sha256(b"LOCAL_CSV_REQUIRED").hexdigest(),
        "note": "Regenerate from local reports/dataset_manifest.csv; JSON is a redacted derivative.",
        "split_reference": split_csv_rel,
        "split_hash": split_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync v4 manifests from canonical CSV sources.")
    parser.add_argument("--dataset-csv", default="reports/dataset_manifest.csv")
    parser.add_argument("--split-csv", default="data/splits/wmh2017_train_val_seed42.csv")
    parser.add_argument("--output-dir", default="artifacts/manifests")
    args = parser.parse_args()

    dataset_csv = (REPO_ROOT / args.dataset_csv).resolve()
    split_csv = (REPO_ROOT / args.split_csv).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv_rel = Path(args.dataset_csv).as_posix()
    split_csv_rel = Path(args.split_csv).as_posix()

    if not split_csv.exists():
        raise SystemExit(f"split CSV not found: {split_csv}")

    split_out = output_dir / "split_manifest.json"
    split_out.write_text(
        json.dumps(_split_manifest_json(split_csv, source_csv_rel=split_csv_rel), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {split_out}")

    if dataset_csv.exists():
        dataset_payload = _dataset_manifest_json(dataset_csv, source_csv_rel=dataset_csv_rel)
    else:
        dataset_payload = _missing_dataset_json(split_csv_rel=split_csv_rel, split_csv=split_csv)
        print("Dataset CSV missing; wrote LOCAL_CSV_REQUIRED placeholder", file=sys.stderr)

    dataset_out = output_dir / "dataset_manifest.json"
    dataset_out.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {dataset_out}")


if __name__ == "__main__":
    main()
