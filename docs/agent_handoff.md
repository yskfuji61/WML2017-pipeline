# Agent Handoff — WMH2017-MONAI-SMOKE-002

task_id: WMH2017-MONAI-SMOKE-002

objective: Build and validate a local MONAI smoke pipeline for WMH2017 using the Dataverse `files` layout, while preventing test leakage and label-policy errors.

operation_class: LOCAL_READ_WRITE_PUBLIC_DATA_NO_CLOUD

release_state: READY_FOR_REQUIREMENTS_REVIEW

## Local dataset root

Use environment variable, not hardcoded code:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Expected:

```text
$WMH2017_ROOT/training
$WMH2017_ROOT/test
$WMH2017_ROOT/additional_annotations
```

## in_scope

- scan `training`, `test`, and `additional_annotations`
- build `reports/dataset_manifest.csv`
- audit training labels
- create train/val split from challenge training only
- mark challenge test as `heldout_eval`
- run MONAI smoke training on training/val only
- record run evidence
- save overlay samples

## out_of_scope

- clinical diagnosis
- customer-facing report
- production deployment
- cloud upload
- private/company medical data
- using challenge test for training/validation/tuning
- using O3/O4 annotations as primary smoke baseline
- official challenge score claim

## target_files

- configs/dataset/wmh2017.yaml
- configs/split/train_val_seed42.yaml
- configs/train/smoke_monai_unet3d.yaml
- src/wmh2017/data/**
- src/wmh2017/evaluation/**
- src/wmh2017/audit/**
- scripts/**
- tests/**
- docs/**
- registry/*_schema.csv

## forbidden_files

- Datasets/**
- **/*.nii
- **/*.nii.gz
- **/*.dcm
- artifacts/checkpoints/**
- artifacts/predictions/**
- any private/PHI data
- raw dataset files

## allowed_commands

```bash
python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
pytest tests/unit
```

## forbidden_commands

```bash
rm -rf "$WMH2017_ROOT"
git add Datasets/
git add '*.nii.gz'
aws s3 cp "$WMH2017_ROOT" ...
gsutil cp "$WMH2017_ROOT" ...
curl -F data=@"$WMH2017_ROOT/..."
```

## required_outputs

- reports/dataset_manifest.csv
- reports/label_value_audit.csv
- data/splits/wmh2017_train_val_seed42.csv
- data/splits/wmh2017_test110_heldout.csv
- data/splits/split_summary.json
- reports/overlays/*.png
- reports/metrics/*.json
- registry/run_manifest.csv or reports/runs/*.json

## validation_commands

```bash
pytest tests/unit
python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
```

## stop_conditions

- `$WMH2017_ROOT` does not contain `training`, `test`, and `additional_annotations`
- expected 60 training / 110 test count fails
- label values outside `{0,1,2}` appear
- label 2 is included as foreground
- test case enters train or validation
- additional observer annotation is used as primary smoke baseline
- raw NIfTI appears in git status
- cloud upload is requested
- private/PHI data is discovered
- result is requested for customer or clinical use

## rollback_plan

- raw dataset is read-only
- generated reports/splits can be deleted and regenerated
- code/config changes reverted by git
- registry changes reviewed before merge

owner: implementation_lead

reviewers:
  - evidence_reviewer
  - medical_domain_reviewer
  - security_privacy_reviewer
