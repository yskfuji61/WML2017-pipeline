# Agent Handoff — WMH2017 SOTA Candidate Audit System

task_id: WMH2017-SOTA-CANDIDATE-AUDIT-001

objective:
  Build an auditable WMH2017 pipeline that first passes EXP-000, then supports winner reproduction and MONAI/nnU-Net comparison without SOTA overclaim.

operation_class:
  LOCAL_READ_WRITE_PUBLIC_DATA_NO_CLOUD

release_state:
  READY_FOR_REQUIREMENTS_REVIEW

in_scope:
  - source register completion
  - dataset card completion
  - manifest generation
  - label audit
  - split generation
  - metric sanity tests
  - 1-case visualization
  - run evidence recording
  - later winner reproduction planning

out_of_scope:
  - clinical diagnosis
  - customer presentation
  - production deployment
  - cloud upload
  - SOTA claim
  - private medical data use
  - model weight publication

target_files:
  - registry/source_register_wmh2017.csv
  - registry/dataset_card_wmh2017.md
  - registry/metric_register_wmh2017.csv
  - registry/split_register_wmh2017.csv
  - registry/experiment_registry_wmh2017.csv
  - registry/claim_boundary_wmh2017.csv
  - registry/failure_taxonomy_wmh2017.csv
  - docs/EXP-000_data_integrity_metric_sanity.md
  - docs/future_sota/EXP-001_winner_reproduction_plan.md
  - src/wmh2017/**
  - scripts/**
  - tests/**

forbidden_files:
  - Datasets/**
  - "**/*.nii"
  - "**/*.nii.gz"
  - "**/*.dcm"
  - artifacts/checkpoints/*
  - artifacts/predictions/*
  - private medical data
  - customer-facing release documents

allowed_commands:
  - python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
  - python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
  - python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
  - python scripts/visualize_wmh_case.py --manifest reports/dataset_manifest.csv --case-id <CASE_ID> --out reports/overlays
  - pytest tests/unit

forbidden_commands:
  - upload dataset to cloud
  - external API calls with raw images
  - rm -rf Datasets/
  - publish model weights
  - package raw NIfTI files
  - use files/test for training validation threshold tuning or model selection

validation_commands:
  - pytest tests/unit
  - python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts

stop_conditions:
  - dataset root not found
  - strict counts fail
  - label values outside {0,1,2}
  - label==2 included as foreground
  - test cases included in training/validation/tuning
  - source license ambiguity blocks reproduction
  - cloud upload required
  - PHI/private data discovered
  - SOTA wording appears before claim review

rollback_plan:
  - raw dataset is read-only
  - generated reports/splits can be deleted and regenerated
  - code/config changes reverted by git
  - registry changes reviewed before merge

owner:
  implementation_lead

reviewers:
  - evidence_reviewer
  - medical_domain_reviewer
  - security_privacy_reviewer
  - ml_reproduction_reviewer
