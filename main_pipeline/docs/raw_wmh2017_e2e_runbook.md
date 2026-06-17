# Raw WMH2017 local E2E runbook

## Purpose

Generate local evidence for the WMH2017 public release without copying raw medical images into this repository.

## Preconditions

- Raw Dataverse files are unpacked locally.
- The path passed to `--files-root` is either the `files/` directory or its parent.
- `requirements-lock.txt` is installed in the active environment.
- Local compute resource is approved for this public-data PoC.
- No proprietary patient data is mixed into this run.

## Audit-only command

```bash
python scripts/run_wmh2017_e2e.py \
  --files-root /path/to/MICCAI2017_WMH/files \
  --work-dir artifacts/runs/wmh2017_audit_only_seed42 \
  --skip-train
```

Expected outputs:

- `dataset_manifest.csv`
- `dataset_manifest.summary.json`
- `label_value_audit.csv`
- `label_value_audit.summary.json`
- `splits/wmh2017_train_val_seed42.csv`
- `splits/wmh2017_test110_heldout.csv`

## Training/evaluation command

```bash
python scripts/run_wmh2017_e2e.py \
  --files-root /path/to/MICCAI2017_WMH/files \
  --run-id wmh2017_local_e2e_seed42 \
  --seed 42 \
  --work-dir artifacts/runs/wmh2017_local_e2e_seed42
```

Expected additional outputs:

- `train/logs/train_log.jsonl`
- `train/checkpoints/model_smoke.pt`
- `train/predictions/*_pred.nii.gz`
- `train/run_evidence.json`
- `metrics/case_metrics.csv`
- `metrics/metrics_summary.json`

## Official evaluator parity

This repository does not vendor or execute the official challenge evaluator. After a human/operator runs the official evaluator and exports case-level metrics:

```bash
python scripts/compare_official_evaluator_parity.py \
  --local artifacts/runs/wmh2017_local_e2e_seed42/metrics/case_metrics.csv \
  --official /path/to/official_case_metrics.csv \
  --out-dir artifacts/runs/wmh2017_local_e2e_seed42/official_parity
```

Expected outputs:

- `official_parity_case_diffs.csv`
- `official_parity_report.json`

## Stop conditions

Stop and request human review when:

- expected counts are not 60 training and 110 test;
- any label audit row reports unexpected values;
- geometry inspection fails;
- train/validation split contains challenge test cases;
- official parity fails;
- cloud compute or proprietary data becomes necessary;
- results are requested for clinical, customer, commercial, or leaderboard claims.
