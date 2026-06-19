#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.audit.run_record import sha256_path
from wmh2017.data.manifest import build_manifest, load_sha256sums

EXPECTED_COUNTS = {
    ("training", "Utrecht"): 20,
    ("training", "Singapore"): 20,
    ("training", "Amsterdam"): 20,
    ("test", "Utrecht"): 30,
    ("test", "Singapore"): 30,
    ("test", "Amsterdam"): 50,
}


def _counts_summary(df):
    if df.empty:
        return {}
    return {
        "by_challenge_split": df.groupby(["challenge_split"]).size().to_dict(),
        "by_challenge_split_site": {
            f"{split}/{site}": int(n)
            for (split, site), n in df.groupby(["challenge_split", "site"]).size().to_dict().items()
        },
        "by_scanner_code": {
            str(scanner): int(n) for scanner, n in df.groupby(["scanner_code"]).size().to_dict().items()
        },
    }


def _validate_expected_counts(df) -> list[str]:
    errors: list[str] = []
    counts = df.groupby(["challenge_split", "site"]).size().to_dict() if not df.empty else {}
    for key, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(key, 0))
        if actual != expected:
            errors.append(f"count mismatch for {key}: expected={expected}, actual={actual}")
    total_training = int((df["challenge_split"] == "training").sum()) if not df.empty else 0
    total_test = int((df["challenge_split"] == "test").sum()) if not df.empty else 0
    if total_training != 60:
        errors.append(f"training total mismatch: expected=60, actual={total_training}")
    if total_test != 110:
        errors.append(f"test total mismatch: expected=110, actual={total_test}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WMH2017 case manifest from the Dataverse `files` root without copying raw data."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Prefer /.../MICCAI2017_WMH/files. Parent containing files/ is also accepted.",
    )
    parser.add_argument("--out", default="registry/dataset_manifest.csv")
    parser.add_argument("--summary-out", default="", help="Optional JSON audit summary path.")
    parser.add_argument(
        "--hash-files", action="store_true", help="Compute sha256 for primary NIfTI files; may be slow."
    )
    parser.add_argument(
        "--sha256sums",
        default="",
        help="Optional SHA256SUMS file from download evidence. Records expected hashes without reading raw NIfTI.",
    )
    parser.add_argument(
        "--inspect-images",
        action="store_true",
        help="Inspect NIfTI shape/spacing/affine metadata for primary FLAIR/T1/WMH files. Requires nibabel.",
    )
    parser.add_argument(
        "--strict-counts", action="store_true", help="Fail unless expected 60 training / 110 test structure is found."
    )
    parser.add_argument(
        "--fail-on-metadata-error",
        action="store_true",
        help="Fail when --inspect-images records any metadata inspection error.",
    )
    args = parser.parse_args()

    sha256sums = load_sha256sums(args.sha256sums) if args.sha256sums else {}
    df = build_manifest(
        Path(args.root),
        hash_files=args.hash_files,
        inspect_images=args.inspect_images,
        sha256sums=sha256sums,
    )
    if df.empty:
        raise SystemExit("No WMH2017 cases found. Check that --root points to the Dataverse `files` directory.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    count_errors = _validate_expected_counts(df)
    metadata_error_cols = [c for c in df.columns if c.endswith("_metadata_status")]
    metadata_errors = []
    for col in metadata_error_cols:
        bad = df[df[col].astype(str).str.startswith("error:", na=False)]
        for _, row in bad.iterrows():
            metadata_errors.append({"case_id": str(row["case_id"]), "column": col, "status": str(row[col])})

    summary = {
        "status": "failed" if count_errors or (args.fail_on_metadata_error and metadata_errors) else "ok",
        "manifest_csv": str(out),
        "manifest_sha256": sha256_path(out),
        "root": str(Path(args.root).expanduser()),
        "hash_files": bool(args.hash_files),
        "inspect_images": bool(args.inspect_images),
        "sha256sums": args.sha256sums,
        "sha256sums_entries": len(sha256sums),
        "n_cases": int(len(df)),
        "counts": _counts_summary(df),
        "count_errors": count_errors,
        "metadata_errors": metadata_errors,
        "critical_rules": [
            "challenge_split=test is held out from train/val/tuning even when labels are present in the 2022 release",
            "additional observer annotations are auxiliary evidence and are not primary training labels by default",
            "raw NIfTI data are referenced by path only and are not copied into the repository",
        ],
    }

    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(df)} case rows to {out}")
    print(f"Wrote dataset audit summary to {summary_out}")
    print("Counts by challenge_split/site:")
    for key, value in sorted(df.groupby(["challenge_split", "site"]).size().to_dict().items()):
        print(f"  {key[0]}/{key[1]}: {value}")

    if args.strict_counts and count_errors:
        raise SystemExit("; ".join(count_errors))
    if args.fail_on_metadata_error and metadata_errors:
        raise SystemExit(f"metadata inspection errors: {len(metadata_errors)}")


if __name__ == "__main__":
    main()
