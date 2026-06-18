# P0/P1 fix report

## Scope

This report records the structural refactor applied after the IA audit.

## Fixed P0 items

| id | fix |
|---|---|
| P0-01 | Canonical MONAI smoke config is now `configs/wmh2017_monai_smoke.yaml`. `README_CURSOR_START.md` points to it. `configs/train/smoke_monai_unet3d.yaml` is retained only as a schema-identical compatibility alias. |
| P0-02 | Source registries no longer contain ambiguous placeholder tokens. Unreviewed fields are explicitly marked as unresolved and blocked from claim use until human review. |
| P0-03 | Experiment registry no longer contains placeholder run IDs or hashes labeled as `UNRESOLVED_PLACEHOLDER`. Pending fields are explicitly marked as generated only after an actual run. |
| P0-04 | Release state is normalized to `READY_FOR_REQUIREMENTS_REVIEW`. Structural checks and preview promotion are separated from release state. |

## Fixed P1 items

| id | fix |
|---|---|
| P1-01 | `README_CURSOR_START.md` is the canonical implementation entry. |
| P1-02 | Future SOTA/winner-reproduction materials moved to `docs/future_sota/`. |
| P1-03 | Config entry is one canonical file plus one compatibility alias. |
| P1-04 | Release decision record states that real-data evidence is not included. |
| P1-05 | `.pytest_cache/` is excluded from the final distribution ZIP. |
| P1-06 | Absolute workstation paths are replaced with `<LOCAL_WMH2017_FILES_ROOT>`. |
| P1-07 | Current working docs are indexed in `docs/README.md`; deferred SOTA docs are separated. |
| P1-08 | Evidence required for preview is listed in `docs/release_decision_record.md`. |

## Remaining non-fixable by AI in this package

- Real WMH2017 dataset manifest cannot be generated without the local dataset.
- Source/license review cannot be completed without human evidence review.
- Official metric parity cannot be claimed without official evaluator verification.
- Clinical/customer/proprietary/cloud approval cannot be granted by this refactor.

## Current state after refactor

```text
release_state: READY_FOR_REQUIREMENTS_REVIEW
preview: blocked_until_real_data_evidence_and_human_review
release: blocked
clinical_customer_use: blocked
```
