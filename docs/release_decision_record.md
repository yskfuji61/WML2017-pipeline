# Release decision record

## Current decision

```text
decision: NOT_APPROVED_FOR_PREVIEW_OR_RELEASE
local_planning_state: READY_FOR_STRUCTURAL_REVIEW
controlled_release_state: NOT_READY_FOR_PREVIEW
package_id: WMH2017-LOCAL-POC-SCAFFOLD
package_version: 0.2.3
package_manifest_sha256: PENDING_GENERATED_MANIFEST
decision_date: 2026-06-17
decision_owner: implementation_lead
release_approver: UNASSIGNED_RELEASE_APPROVER
linked_manifest: reports/full_package_manifest.json
linked_findings: registry/finding_register_wmh2017.csv
```

## Decision boundary

This package contains executable scaffolding, local unit tests, deterministic
file layout expectations, CI workflow definitions, evidence commands, and a
MONAI smoke-training entry point.

It is not `READY_FOR_PREVIEW`, `READY_FOR_LIMITED_INTERNAL_USE`, or
`READY_FOR_RELEASE`.

`READY_FOR_REQUIREMENTS_REVIEW` is a local planning state only. See
`docs/release_state_crosswalk.md`.

## Why this is not preview-ready

- Source/license review is incomplete.
- No real WMH2017 dataset manifest is included.
- No target-machine real-data run log is included.
- No real checkpoint hash or prediction artifact from WMH2017 data is included.
- No official WMH challenge evaluation-code parity record is included.
- No reviewer approval record is included.
- No release approver exists for this package.
- No security/privacy approval exists for proprietary, patient, customer, or cloud data.
- No clinical, regulatory/QMS, customer, or production review exists.
- Open Sev1 findings remain in `registry/finding_register_wmh2017.csv`.

## Structural evidence now required

The structural package identity is recorded in:

```text
reports/full_package_manifest.json
```

This manifest is file identity evidence only. It is not proof of WMH2017
training, inference, evaluation, source review, or approval.

## Promotion requirements to reach READY_FOR_PREVIEW

Attach all of the following and close related Sev1 findings:

1. Completed source/license review in `registry/source_register_wmh2017.csv`.
2. `reports/dataset_manifest.csv` and sha256.
3. `reports/label_value_audit.csv`.
4. `data/splits/wmh2017_train_val_seed42.csv` and sha256.
5. At least one overlay image and reviewer note.
6. `artifacts/runs/<run_id>/logs/train_log.jsonl`.
7. `artifacts/runs/<run_id>/run_evidence.json`.
8. `artifacts/runs/<run_id>/predictions/*_pred.nii.gz` and sha256.
9. `artifacts/runs/<run_id>/evaluation/case_metrics.csv`.
10. `artifacts/runs/<run_id>/evaluation/metrics_summary.json`.
11. `registry/run_manifest.csv`.
12. Local command transcript or CI log.
13. Review disposition in `registry/review_approval_register_wmh2017.csv`.
14. Claim-boundary confirmation that no clinical, customer, production, or SOTA claim is made.

## Promotion requirements beyond preview

Before any customer-facing, clinical, proprietary-data, cloud, production,
official benchmark, or SOTA use, route to the appropriate human reviews:

- Dataset Governance
- Evidence/Source Review
- Security/Privacy Review
- Medical/Domain Review
- Model Validation
- Regulatory/QMS Review
- Legal/Contract Review
- Release Approval

## Explicit non-approval

The following remain blocked:

```text
clinical_use
diagnostic_use
reader_replacement
patient_risk_judgment
customer_presentation
production_deployment
proprietary_data_processing
unapproved_cloud_upload
official_benchmark_equivalence
sota_or_leaderboard_claim
```

## Reversal conditions

This non-release decision may only be reconsidered after all open Sev1 findings
are closed with evidence and human reviewer disposition.


## Update: official public download evidence attached

As of `2026-06-18`, the package includes non-raw download evidence under:

```text
evidence/wmh2017_download_2026-06-16/
```

This evidence records:

- DOI `10.34894/AECRSD`
- Dataverse release/publication date `2022-12-21`
- license value reported by Dataverse metadata: `CC-BY-NC-4.0`
- local download timestamp `2026-06-16T21:49:48Z`
- `1791` downloaded-file manifest entries
- `1791` SHA256 entries
- SHA256 verification log with all entries marked `OK`
- file size check matching Dataverse metadata

This changes the status of the dataset-acquisition evidence only. It does not
approve source/license interpretation, commercial/customer use, clinical use,
training/evaluation claims, official benchmark comparability, Preview, or
Release.
