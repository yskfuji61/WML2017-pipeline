#!/usr/bin/env bash
set -euo pipefail

: "${WMH2017_ROOT:?Set WMH2017_ROOT to the local MICCAI2017_WMH/files directory or its parent. Raw data must stay outside git.}"

python -m pytest -q

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

python scripts/visualize_wmh_case.py \
  --manifest reports/dataset_manifest.csv \
  --case-id "$(python - <<'PY'
import pandas as pd
df=pd.read_csv('reports/dataset_manifest.csv')
print(df[df.challenge_split.astype(str).str.lower().eq('training')].iloc[0]['case_id'])
PY
)" \
  --out reports/overlays

python scripts/train_wmh2017.py \
  --config configs/wmh2017_monai_smoke.yaml

python scripts/evaluate_wmh2017.py \
  --manifest reports/dataset_manifest.csv \
  --split data/splits/wmh2017_train_val_seed42.csv \
  --predictions artifacts/runs/wmh2017_monai_smoke_seed42/predictions \
  --out-dir artifacts/runs/wmh2017_monai_smoke_seed42/evaluation \
  --run-id wmh2017_monai_smoke_seed42 \
  --assigned-split val \
  --threshold 0.5

python scripts/verify_release_package.py --repo-root . --out reports/release_package_manifest.json
