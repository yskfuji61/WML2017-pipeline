# Top-pro refactor implementation report v0.2.2

Generated: 2026-06-18

## Scope

This patch adds executable scaffolding for the remaining high-risk evidence gaps:

1. Raw WMH2017 `files/` root dataset manifest generation.
2. Real NIfTI label audit.
3. Real MONAI training entrypoint integration.
4. Prediction persistence.
5. Validation Dice/HD95/AVD/lesion metrics.
6. Official evaluator parity comparison against a supplied official export.

Raw medical images are not included in this repository package.

## Implemented changes

### Dataset manifest

`src/wmh2017/data/manifest.py` and `scripts/audit_wmh2017_dataset.py` now support:

- root autodetection for `files/` or parent directory;
- expected SHA256 recording from `SHA256SUMS.txt` without reading raw files;
- optional raw SHA256 hashing when explicitly requested;
- optional NIfTI metadata inspection for FLAIR/T1/WMH files;
- strict expected-count checks for 60 training and 110 test cases;
- JSON summary output with count and metadata errors.

### Label audit

`scripts/audit_wmh2017_labels.py` now supports:

- training/test/all selection;
- geometry metadata capture for audited masks;
- JSON summary output;
- fail-closed mode on missing masks or invalid label values;
- explicit label policy: `label == 1` is WMH foreground and `label == 2` is not foreground.

### End-to-end local evidence chain

`scripts/run_wmh2017_e2e.py` orchestrates:

1. dataset manifest;
2. label audit;
3. train/validation split;
4. MONAI smoke training and prediction save;
5. local validation evaluation;
6. optional official evaluator parity comparison.

This is an operator convenience script. Each underlying step remains independently callable and auditable.

### Official evaluator parity

`src/wmh2017/evaluation/official_parity.py` and `scripts/compare_official_evaluator_parity.py` compare local case-level metrics with a supplied official evaluator CSV/TSV/JSON export. The repository does not vendor or execute official challenge code. This avoids unreviewed third-party code execution while still creating a hard gate for leaderboard-comparable claims.

## Validation run in this environment

```text
python -m compileall scripts src tests
status: passed

python -m pytest -q
status: passed
tests: 36 passed
```

A Python startup warning from the hosted spreadsheet runtime appeared on stderr. It is outside this repository and did not change the compile or pytest exit status.

## Not executed in this environment

- Real 8.7GB raw `files/` manifest generation.
- Real NIfTI label audit.
- MONAI training.
- Prediction generation.
- Validation Dice computation on raw data.
- Official evaluator parity against an actual official export.

Reason: the raw WMH2017 `files/` tree and an official evaluator export are not present inside this execution sandbox.

## Required operator commands

Assuming raw data is unpacked at `/path/to/MICCAI2017_WMH/files`:

```bash
python scripts/run_wmh2017_e2e.py \
  --files-root /path/to/MICCAI2017_WMH/files \
  --run-id wmh2017_local_e2e_seed42 \
  --seed 42 \
  --work-dir artifacts/runs/wmh2017_local_e2e_seed42
```

For audits only:

```bash
python scripts/run_wmh2017_e2e.py \
  --files-root /path/to/MICCAI2017_WMH/files \
  --work-dir artifacts/runs/wmh2017_audit_only_seed42 \
  --skip-train
```

For official parity once an official evaluator export exists:

```bash
python scripts/compare_official_evaluator_parity.py \
  --local artifacts/runs/wmh2017_local_e2e_seed42/metrics/case_metrics.csv \
  --official /path/to/official_case_metrics.csv \
  --out-dir artifacts/runs/wmh2017_local_e2e_seed42/official_parity
```

## Release state

Still not release-ready.

Current state: `READY_FOR_REQUIREMENTS_REVIEW`.

The code is better prepared for real evidence generation, but no clinical/customer/commercial/release claim is allowed until the raw-data run evidence, official parity, source/license review, security/privacy review, and domain review are completed.
