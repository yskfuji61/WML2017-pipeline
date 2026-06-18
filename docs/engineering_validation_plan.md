# Engineering validation plan

This repository is a local research PoC scaffold for WMH2017 white matter hyperintensity segmentation.
It is not a diagnostic, clinical, customer-facing, cloud, or production system.

## Acceptance evidence

| AC | Evidence command | Expected artifact | Release implication |
|---|---|---|---|
| AC-01 dataset acquired/unpacked | `python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts` | `reports/dataset_manifest.csv` | Required before training |
| AC-02 image/label readable | `python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training` | `reports/label_value_audit.csv` | Required before training |
| AC-03 overlay created | `python scripts/visualize_wmh_case.py --manifest reports/dataset_manifest.csv --case-id <training_case>` | `reports/overlays/*_overlay.png` | Human review gate |
| AC-04 train/val split | `python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42` | `data/splits/wmh2017_train_val_seed42.csv` | Test110 contamination blocked |
| AC-05 training executes | `python scripts/train_wmh2017.py --config configs/wmh2017_monai_smoke.yaml` | train log, checkpoint, run manifest | Smoke only, not performance claim |
| AC-06 prediction mask saved | same as AC-05 | `artifacts/runs/<run_id>/predictions/*_pred.nii.gz` | Validation prediction evidence |
| AC-07 metrics calculated | `python scripts/evaluate_wmh2017.py ...` | `case_metrics.csv`, `metrics_summary.json` | Local validation only |
| AC-08 experiment conditions recorded | AC-05/AC-07 | `registry/run_manifest.csv`, `run_evidence.json` | Audit trace |
| AC-09 discrepancy analysis | manual review after AC-07 | `docs/experiment_notes/<run_id>.md` | Required before comparison claims |
| AC-10 no unauthorized data/cloud/customer use | review checklist | `docs/release_decision_record.md` | Blocks external use |

## Minimal full local command

```bash
bash scripts/run_wmh2017_minimal_pipeline.sh
```

## Stop conditions

- Raw data appears inside git-controlled `data/`, `artifacts/`, or `reports/`.
- `challenge_split=test` is used for training, validation, early stopping, threshold tuning, preprocessing fit, or model selection.
- Label value 2 is treated as foreground.
- Cloud execution or proprietary data becomes necessary before written approval.
- Metrics are requested for customer, diagnostic, clinical, leaderboard, or SOTA claims before official evaluation cross-check and human review.


## Evidence ownership and pass/fail rule

Each AC must be recorded with:

- owner
- command
- exact input paths
- output artifact path
- output sha256
- pass/fail result
- reviewer disposition, if human review is required
- release implication
- linked finding ID if failed or incomplete

An AC is not closed by code presence alone. It is closed only by generated
artifact evidence or explicit reviewer disposition.

## Minimum run evidence fields

`run_evidence.json` must include:

```text
run_id
timestamp
operator
machine_class
python_version
dependency_versions
git_commit_or_package_hash
config_path
config_sha256
dataset_manifest_path
dataset_manifest_sha256
split_manifest_path
split_manifest_sha256
seed
device
device_requested
device_selected
mps_available
mps_convtranspose_patched
mps_convtranspose_replaced_count
mps_fallback_enabled
model_patch
patch_scope
native_mps_claim
model_architecture
checkpoint_path
checkpoint_sha256
prediction_paths
prediction_sha256
metric_script_path
metric_script_sha256
case_metrics_path
case_metrics_sha256
metrics_summary_path
metrics_summary_sha256
claim_boundary
known_limitations
stop_conditions_checked
```
