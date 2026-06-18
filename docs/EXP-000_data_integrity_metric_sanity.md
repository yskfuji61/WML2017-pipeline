# EXP-000 Protocol — Data Integrity + Metric Sanity + 1-Case Visualization

## Objective

Verify the local WMH2017 dataset layout, label policy, split guard, and metric sanity before any training or SOTA-oriented work.

## Required commands

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"

python scripts/audit_wmh2017_dataset.py \
  --root "$WMH2017_ROOT" \
  --out reports/dataset_manifest.csv \
  --strict-counts

python scripts/audit_wmh2017_labels.py \
  --manifest reports/dataset_manifest.csv \
  --split training \
  --out reports/label_value_audit.csv

python scripts/make_wmh2017_splits.py \
  --manifest reports/dataset_manifest.csv \
  --seed 42 \
  --out-dir data/splits

pytest tests/unit
```

## Required outputs

- `reports/dataset_manifest.csv`
- `reports/label_value_audit.csv`
- `data/splits/wmh2017_train_val_seed42.csv`
- `data/splits/wmh2017_test110_heldout.csv`
- `data/splits/split_summary.json`
- at least one overlay in `reports/overlays/`
- run record in `reports/runs/` or `registry/run_manifest.csv`

## Pass criteria

- training=60 and test=110 under strict counts,
- no labels outside `{0,1,2}` in training masks,
- label==2 is not foreground,
- test rows are heldout only,
- DSC/H95/AVD/lesion recall/lesion F1 sanity tests pass.

## Stop conditions

- dataset root not found,
- second training path should have been test path and cannot be verified,
- strict counts fail,
- label values outside `{0,1,2}`,
- test rows enter train/val,
- cloud upload or external API is required.
