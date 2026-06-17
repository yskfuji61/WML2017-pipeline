# Package control index

## Scope

Controlled index for the WMH2017 local research PoC scaffold. This is not a
clinical, customer, proprietary-data, cloud, production, official benchmark, or
SOTA release package.

## Artifact identity

| Field | Value |
|---|---|
| package_name | wmh2017-pipeline |
| package_version | 0.2.0 |
| package_status | REQUIREMENTS_REVIEW_ONLY |
| controlled_release_state | NOT_READY_FOR_PREVIEW |
| source_package | wmh2017_pipeline_refactored_toppro_qc.zip, refactored into this package |
| effective_date | NOT_EFFECTIVE_FOR_RELEASE |
| owner | implementation_lead |
| reviewer | UNASSIGNED_HUMAN_REVIEWER |
| approver | UNASSIGNED_RELEASE_APPROVER |
| approval_status | NOT_APPROVED |
| DLP class | Public scaffold / no raw medical images bundled |
| data boundary | public WMH2017 data may be read locally only after source/license review |
| retention | keep generated run evidence under `artifacts/runs/<run_id>/`; do not commit raw images or PHI |
| rollback route | restore previous package hash or revert git commit after failed structural/evidence review |

## Current authoritative entry points

1. `README_CURSOR_START.md`
2. `docs/release_state_crosswalk.md`
3. `docs/engineering_validation_plan.md`
4. `docs/release_decision_record.md`
5. `docs/final_evidence_binder_index.md`
6. `registry/finding_register_wmh2017.csv`
7. `reports/full_package_manifest.json`

## Controlled document rule

A file is not release-controlled merely because it exists in this folder. A file
is controlled only when it appears in `reports/full_package_manifest.json` and,
for release-relevant files, is linked to owner, status, source/review boundary,
finding impact, and release implication.

## Non-controlled / inherited areas

`core/pipeline/**` is inherited reference material. It is not the primary
authoritative path for the June MONAI smoke pipeline unless a later review record
explicitly promotes a file from that area.

The current authoritative implementation path is:

```text
src/wmh2017/**
scripts/audit_wmh2017_dataset.py
scripts/audit_wmh2017_labels.py
scripts/make_wmh2017_splits.py
scripts/visualize_wmh_case.py
scripts/train_wmh2017.py
scripts/evaluate_wmh2017.py
configs/wmh2017_monai_smoke.yaml
tests/unit/**
```
