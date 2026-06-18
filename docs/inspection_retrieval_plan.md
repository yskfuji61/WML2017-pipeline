# Inspection retrieval plan

## Purpose

Define how a reviewer can retrieve evidence without relying on oral explanation,
folder names, screenshots, or local memory.

## Retrieval owner

| Role | Current assignment | SLA | Status |
|---|---|---|---|
| implementation evidence owner | implementation_lead | next working session | tentative |
| source/license reviewer | UNASSIGNED_HUMAN_REVIEWER | not set | open |
| security/privacy reviewer | UNASSIGNED_HUMAN_REVIEWER | not set | open |
| model validation reviewer | UNASSIGNED_HUMAN_REVIEWER | not set | open |
| release approver | UNASSIGNED_RELEASE_APPROVER | not set | open |

## Retrieval map

| Question | Evidence path | Required before |
|---|---|---|
| What package was reviewed? | `reports/full_package_manifest.json` | structural review |
| What source terms allow this? | `registry/source_register_wmh2017.csv`, `docs/source_license_review_checklist.md` | data use beyond local inspection |
| What data was used? | `reports/dataset_manifest.csv` | training/evaluation |
| Were labels sane? | `reports/label_value_audit.csv` | training |
| What split was used? | `data/splits/wmh2017_train_val_seed42.csv` | training/evaluation |
| What command ran? | `artifacts/runs/<run_id>/run_evidence.json` | any metric claim |
| What was predicted? | `artifacts/runs/<run_id>/predictions/` + hashes | metric claim |
| What metrics were calculated? | `artifacts/runs/<run_id>/evaluation/` | local validation claim |
| What was reviewed? | `registry/review_approval_register_wmh2017.csv` | preview/release |
| What blocks promotion? | `registry/finding_register_wmh2017.csv` | all gates |

## Confidentiality rule

Do not provide raw images, patient-like metadata, private paths, or proprietary
data to reviewers unless the DLP class and access authorization are recorded.
