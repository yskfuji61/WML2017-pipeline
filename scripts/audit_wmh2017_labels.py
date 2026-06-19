#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.audit.run_record import sha256_path
from wmh2017.data.label_policy import ALLOWED_LABELS, audit_mask
from wmh2017.io.images import load_image_metadata


def load_mask(path: str):
    try:
        import nibabel as nib

        return nib.load(path).get_fdata()
    except ImportError as e:
        raise SystemExit("nibabel is required for label audit. Install nibabel or add SimpleITK loader.") from e


def _summary(rows: list[dict], output_csv: Path) -> dict:
    df = pd.DataFrame(rows)
    status_counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    return {
        "status": "ok"
        if not any(str(k).startswith("error") or k == "missing_mask" for k in status_counts)
        else "failed",
        "label_audit_csv": str(output_csv),
        "label_audit_sha256": sha256_path(output_csv) if output_csv.exists() else "",
        "n_rows": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "total_foreground_voxels": int(ok["foreground_voxels"].sum()) if not ok.empty else 0,
        "total_ignore_voxels": int(ok["ignore_voxels"].sum()) if not ok.empty else 0,
        "cases_with_label2": int((ok["has_label2"].astype(str).str.lower() == "true").sum()) if not ok.empty else 0,
        "allowed_labels": sorted(int(x) for x in ALLOWED_LABELS),
        "label_policy": "foreground is label==1 only; label==2 is ignored as foreground and must not be converted by mask>0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WMH2017 label values with label==1 foreground policy.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="reports/label_value_audit.csv")
    parser.add_argument("--summary-out", default="", help="Optional JSON summary output path.")
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "test", "all"],
        help="Default is training. Test labels, if present in local 2022 release, must not be used for train/val/tuning.",
    )
    parser.add_argument(
        "--include-geometry",
        action="store_true",
        help="Record shape/spacing/affine metadata for audited masks. Requires nibabel.",
    )
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero on missing masks or audit errors.")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_col = "challenge_split" if "challenge_split" in manifest.columns else "source_split"
    if args.split != "all" and split_col in manifest.columns:
        manifest = manifest[manifest[split_col].astype(str).str.lower() == args.split].copy()

    rows = []
    for _, r in manifest.iterrows():
        mask_path = str(r.get("wmh_path", "") or r.get("mask_path", "") or "")
        row_base = {
            "case_id": r["case_id"],
            "challenge_split": r.get("challenge_split", r.get("source_split", "")),
            "site": r.get("site", ""),
            "scanner_code": r.get("scanner_code", ""),
            "mask_path": mask_path,
            "mask_sha256": r.get("mask_sha256", ""),
            "mask_expected_sha256": r.get("mask_expected_sha256", ""),
        }
        if not mask_path:
            rows.append(
                {
                    **row_base,
                    "status": "missing_mask",
                    "values": "",
                    "has_label2": "",
                    "foreground_voxels": "",
                    "ignore_voxels": "",
                    "shape": "",
                    "spacing": "",
                    "affine_sha256": "",
                }
            )
            continue
        try:
            audit = audit_mask(load_mask(mask_path))
            geometry = {"shape": "", "spacing": "", "affine_sha256": ""}
            if args.include_geometry:
                meta = load_image_metadata(mask_path)
                geometry = {
                    "shape": "x".join(str(x) for x in meta.shape),
                    "spacing": "x".join(f"{float(x):.8g}" for x in meta.spacing),
                    "affine_sha256": meta.affine_sha256,
                }
            rows.append(
                {
                    **row_base,
                    "status": "ok",
                    "values": "|".join(map(str, audit.values)),
                    "has_label2": audit.has_label2,
                    "foreground_voxels": audit.foreground_voxels,
                    "ignore_voxels": audit.ignore_voxels,
                    **geometry,
                }
            )
        except Exception as e:
            rows.append(
                {
                    **row_base,
                    "status": f"error:{type(e).__name__}",
                    "values": str(e),
                    "has_label2": "",
                    "foreground_voxels": "",
                    "ignore_voxels": "",
                    "shape": "",
                    "spacing": "",
                    "affine_sha256": "",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    summary = _summary(rows, out)
    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote label audit to {out}")
    print(f"Wrote label audit summary to {summary_out}")
    if args.fail_on_error and summary["status"] != "ok":
        raise SystemExit(f"label audit failed: {summary['status_counts']}")


if __name__ == "__main__":
    main()
