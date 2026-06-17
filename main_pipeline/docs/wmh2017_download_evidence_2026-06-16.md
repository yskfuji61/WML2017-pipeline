# WMH2017 Download Evidence — 2026-06-16

## Status

Evidence captured. Human source/license review is still required before any
source-dependent external, customer, commercial, official benchmark, or release
claim.

This directory contains download metadata and checksum evidence only. It does
not contain raw NIfTI medical image data.

## Evidence directory

```text
evidence/wmh2017_download_2026-06-16/
```

Included files:

- `dataverse_metadata.json`
- `download_manifest.tsv`
- `download_record.txt`
- `downloaded_file_manifest.txt`
- `SHA256SUMS.txt`
- `sha256_verify.log`
- `readme.pdf`
- `evidence_file_manifest.csv`

## Captured dataset identity

- Dataset: MICCAI 2017 WMH / White Matter Hyperintensity Segmentation Challenge
- Source DOI: `10.34894/AECRSD`
- Persistent URL: `https://doi.org/10.34894/AECRSD`
- Dataverse publication date: `2022-12-21`
- Downloaded at: `2026-06-16T21:49:48Z`
- License reported by Dataverse metadata: `CC-BY-NC-4.0`

## Captured local acquisition evidence

From `download_record.txt`:

- downloaded file count: `1791`
- SHA256 entry count: `1791`
- total downloaded size: `8.7G`
- file size check: `All file sizes match Dataverse metadata.`

From `sha256_verify.log`:

- all `1791` listed entries end with `: OK`.

## Verification command

```bash
python scripts/verify_wmh2017_download_evidence.py   --evidence-dir evidence/wmh2017_download_2026-06-16   --out reports/wmh2017_download_evidence_verification.json
```

## Boundary

This evidence closes only the question: "do we have a structured record that the
official public dataset was downloaded and checksum-verified locally?"

It does not close:

- source/license review
- terms-of-use review
- real dataset manifest generation from the local `files/` root
- label audit
- train/validation split
- MONAI smoke training
- prediction output
- local metric output
- official evaluator parity
- medical review
- security/privacy review
- release or preview approval
