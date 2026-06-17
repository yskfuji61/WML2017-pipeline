#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_EVIDENCE_FILES = {
    "dataverse_metadata.json",
    "download_manifest.tsv",
    "download_record.txt",
    "downloaded_file_manifest.txt",
    "SHA256SUMS.txt",
    "sha256_verify.log",
    "readme.pdf",
}

RAW_MEDICAL_SUFFIXES = (".nii", ".nii.gz")


def _non_empty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def verify_download_evidence(evidence_dir: str | Path) -> dict:
    root = Path(evidence_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"evidence directory not found: {root}")

    missing = sorted(name for name in REQUIRED_EVIDENCE_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"missing required evidence files: {missing}")

    raw_entries = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith(RAW_MEDICAL_SUFFIXES)
    ]
    if raw_entries:
        raise ValueError(f"raw medical image files must not be packaged as evidence: {raw_entries[:10]}")

    downloaded = _non_empty_lines(root / "downloaded_file_manifest.txt")
    sha_entries = _non_empty_lines(root / "SHA256SUMS.txt")
    verify_lines = _non_empty_lines(root / "sha256_verify.log")
    failed_verify = [line for line in verify_lines if not line.endswith(": OK")]

    record = (root / "download_record.txt").read_text(encoding="utf-8", errors="replace")
    required_record_phrases = [
        "Source DOI: 10.34894/AECRSD",
        "Number of downloaded files:",
        "Number of SHA256 entries:",
        "All file sizes match Dataverse metadata.",
    ]
    missing_record_phrases = [phrase for phrase in required_record_phrases if phrase not in record]
    if missing_record_phrases:
        raise ValueError(f"download_record.txt is missing required phrases: {missing_record_phrases}")

    if len(downloaded) != len(sha_entries):
        raise ValueError(f"downloaded_file_manifest/SHA256SUMS count mismatch: {len(downloaded)} != {len(sha_entries)}")

    if len(verify_lines) != len(sha_entries):
        raise ValueError(f"sha256_verify.log/SHA256SUMS count mismatch: {len(verify_lines)} != {len(sha_entries)}")

    if failed_verify:
        raise ValueError(f"sha256 verification failures found: {failed_verify[:10]}")

    return {
        "status": "passed",
        "evidence_dir": str(root),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_file_count": len(downloaded),
        "sha256_entry_count": len(sha_entries),
        "sha256_verified_ok_count": len(verify_lines),
        "raw_medical_files_in_evidence_package": 0,
        "claim_boundary": (
            "Download evidence verifies local acquisition metadata and checksum logs only; "
            "it is not source/license approval, real training evidence, evaluation evidence, "
            "clinical validation, customer approval, or release approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify packaged WMH2017 download evidence without raw medical images.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = verify_download_evidence(args.evidence_dir)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
