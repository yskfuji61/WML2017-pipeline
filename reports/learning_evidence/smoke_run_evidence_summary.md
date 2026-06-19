# Smoke run evidence summary (audit refactor)

Last updated: 2026-06-19

## Claim boundary

- Local WMH2017 PoC smoke wiring validation only.
- Not clinical, customer, production, SOTA, or official benchmark evidence.
- Local validation metrics are not hidden-test leaderboard results.

## Run identity

| field | value |
|---|---|
| run_id | `wmh2017_smoke_refactor_20260619` |
| config | `configs/wmh2017_monai_smoke.yaml` |
| split | `data/splits/wmh2017_train_val_seed42.csv` (seed=42) |
| WMH2017_ROOT | `<LOCAL_WMH2017_FILES_ROOT>` |
| device | auto (MPS patch path on Apple Silicon when selected) |

## Prior validated smoke reference (seed42, wiring check)

From local PoC run EXP-000; reproduced pipeline behavior before artifact expiry on disk.

| metric | value |
|---|---:|
| n_cases (val with predictions) | 2 |
| mean_dice | 0.00105 |
| mean_hd95 | 127.58 |
| mean_lesion_recall | 0.931 |
| mean_lesion_f1 | 0.024 |
| global_step | 2 |

**Interpretation:** Near-zero Dice is expected for a 2-step smoke run with 32³ patches; confirms train→infer→metric wiring, not model performance.

## Artifacts policy

- Checkpoints, predictions (`.nii.gz`), and raw `case_metrics.csv` remain **gitignored** under `artifacts/runs/`.
- Hashes and status are recorded in `registry/runs/run_manifest.csv` after a local rerun.
- Pred-vs-GT overlays are generated locally via `scripts/visualize_wmh_case.py --prediction ...` into gitignored `reports/overlays/`.

## Local rerun command

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
make e2e RUN_ID=wmh2017_smoke_refactor_20260619 WMH2017_ROOT="$WMH2017_ROOT"
```

Then evaluate and update this summary with fresh redacted metrics only.
