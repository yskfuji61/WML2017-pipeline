# Cursor Start — WMH2017 local smoke pipeline

## Current state

```text
local_planning_state: READY_FOR_REQUIREMENTS_REVIEW
controlled_release_state: NOT_READY_FOR_PREVIEW
structural_checks: unit_tests_partial_passed
real_data_run: not_verified_in_package
clinical_customer_release: blocked
```

This repository is for the June catch-up phase: verify that public WMH2017 data can be loaded, split, visualized, trained on briefly with a MONAI/PyTorch 3D segmentation model, inferred, and evaluated locally.

It is not a clinical, customer-facing, production, diagnostic, or SOTA-claim package.


## Mandatory audit registers before promotion

Do not promote this repository based on narrative quality. Use these files:

```text
reports/full_package_manifest.json
registry/finding_register_wmh2017.csv
registry/review_approval_register_wmh2017.csv
registry/decision_record_register_wmh2017.csv
docs/final_evidence_binder_index.md
docs/release_state_crosswalk.md
docs/source_license_review_checklist.md
docs/security_privacy_gate.md
docs/model_validation_protocol_wmh2017.md
docs/official_metric_parity_plan_wmh2017.md
docs/rollback_plan_wmh2017.md
docs/inspection_retrieval_plan.md
```

All generated metrics must be linked to run_id, config hash, dataset manifest
hash, split manifest hash, prediction hash, metric script hash, and claim
boundary. Metrics without this linkage are not valid evidence.

## Single active route

Use this route only for the first pass:

```text
EXP-000 data integrity + metric sanity + 1-case visualization
→ MONAI 3D U-Net smoke training
→ validation prediction + Dice/evidence record
```

Future SOTA-candidate materials are parked under `docs/future_sota/` and must not be used before EXP-000 evidence exists.

## Local data root

Set `WMH2017_ROOT` to the local Dataverse `files` directory. Do not commit absolute workstation paths.

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Expected directories:

```text
$WMH2017_ROOT/training
$WMH2017_ROOT/test
$WMH2017_ROOT/additional_annotations
```

If a prompt or note shows two `training` paths, treat the second as a possible typo and verify whether `$WMH2017_ROOT/test` exists before scanning test data.

## Required first commands

```bash
python scripts/audit_wmh2017_dataset.py   --root "$WMH2017_ROOT"   --out reports/dataset_manifest.csv   --strict-counts

python scripts/audit_wmh2017_labels.py   --manifest reports/dataset_manifest.csv   --split training   --out reports/label_value_audit.csv

python scripts/make_wmh2017_splits.py   --manifest reports/dataset_manifest.csv   --seed 42   --out-dir data/splits

pytest tests/unit
```

## MONAI smoke training

Run this only after the manifest, label audit, split, and unit tests pass.

```bash
python scripts/train_wmh2017.py --config configs/wmh2017_monai_smoke.yaml
```

`configs/wmh2017_monai_smoke.yaml` is the canonical smoke config. `configs/train/smoke_monai_unet3d.yaml` is retained as a compatibility alias with the same schema.

## Done definition before July

- WMH2017 source, license, and use boundary are recorded.
- `reports/dataset_manifest.csv` is generated.
- `reports/label_value_audit.csv` is generated.
- At least one FLAIR + label==1 overlay is saved under `reports/overlays/`.
- `data/splits/wmh2017_train_val_seed42.csv` is generated.
- Unit tests pass with the exact command logged.
- MONAI 3D U-Net smoke training completes.
- Validation prediction masks are saved.
- Validation Dice or equivalent local metric is computed.
- `run_evidence.json`, logs, config path, dataset manifest path, split manifest path, seed, device, and claim boundary are recorded.
- No proprietary data, unapproved cloud upload, customer presentation, clinical decision support, or production claim is made.

## Stop conditions

Stop and ask for human review if any of the following occurs:

- dataset root cannot be verified,
- label values fall outside the expected policy,
- test rows enter train/validation/threshold tuning,
- cloud upload is required,
- proprietary or patient-identifiable data may be involved,
- official challenge or SOTA comparison is requested before source/metric parity review,
- results are unexpectedly low or high and the cause is not understood.

## Files to edit first

```text
configs/wmh2017_monai_smoke.yaml
scripts/audit_wmh2017_dataset.py
scripts/audit_wmh2017_labels.py
scripts/make_wmh2017_splits.py
scripts/visualize_wmh_case.py
scripts/train_wmh2017.py
scripts/evaluate_wmh2017.py
src/wmh2017/
tests/unit/
```

Do not make broad "fix all related files" edits without a file list, command list, rollback plan, and stop conditions.
