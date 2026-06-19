# WMH2017 split policy (v4)

- Train/val split uses challenge **training** cases only (`challenge_split=training`).
- Test/hidden-test cases are held out (`heldout_eval`) and must not appear in train or val.
- **Canonical split**: `data/splits/wmh2017_train_val_seed42.csv` (seed=42, train 48 / val 12).
- Test split must not be used for training, validation, threshold tuning, model selection, or early stopping.
- Split overlap verification: `scripts/data/verify_no_split_overlap.py`.
- Regenerate v4 JSON derivative: `scripts/data/sync_v4_manifests_from_csv.py`.
