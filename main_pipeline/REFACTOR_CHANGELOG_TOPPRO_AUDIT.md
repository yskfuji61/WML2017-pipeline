# Refactor changelog — toppro audit hardening

## Refactor boundary

This refactor hardens the package as a local public-data WMH2017 PoC scaffold.
It does not fabricate real WMH2017 data evidence, training logs, predictions,
metrics, source/license approval, clinical review, customer approval, or release
approval.

## Implemented changes

1. Added controlled release-state crosswalk to prevent `READY_FOR_REQUIREMENTS_REVIEW`
   from being mistaken for preview or release readiness.
2. Added package control index and clarified authoritative current path versus
   inherited reference path.
3. Replaced partial compatibility manifest with a full structural manifest
   generator and generated `reports/full_package_manifest.json`.
4. Added open finding register with Sev1/Sev2/Sev3 items and explicit closure
   evidence.
5. Added review/approval register with unassigned reviewer/approver status and
   exact non-approval boundaries.
6. Added decision record register and updated release decision record to
   non-approval.
7. Added final evidence binder index separating scaffold files from real run
   evidence.
8. Added source/license review checklist and extended source register fields.
9. Added model validation protocol and official metric parity plan.
10. Added security/privacy gate, rollback plan, and inspection retrieval plan.
11. Added run evidence schema.
12. Updated README and Cursor entrypoint to make blocked claims explicit.
13. Updated audit map to reduce inherited-file navigation ambiguity.
14. Ran structural manifest generation and pytest validation.

## Files added

- `docs/release_state_crosswalk.md`
- `docs/package_control_index.md`
- `docs/final_evidence_binder_index.md`
- `docs/source_license_review_checklist.md`
- `docs/model_validation_protocol_wmh2017.md`
- `docs/security_privacy_gate.md`
- `docs/rollback_plan_wmh2017.md`
- `docs/inspection_retrieval_plan.md`
- `docs/official_metric_parity_plan_wmh2017.md`
- `registry/finding_register_wmh2017.csv`
- `registry/review_approval_register_wmh2017.csv`
- `registry/decision_record_register_wmh2017.csv`
- `registry/run_evidence_schema_wmh2017.csv`
- `reports/static_refactor_validation_report.json`
- `reports/full_package_manifest.json`
- `docs/experiment_notes/.gitkeep`

## Validation result

```text
python -m pytest tests/unit tests/integration
24 passed
```

## Remaining open state

The package remains not preview-ready and not release-ready until source review,
real public-data run evidence, official metric parity, human review disposition,
and release decision are completed.


## 2026-06-18 — P0 evidence/geometry/parity hardening

### Changed
- Added `src/wmh2017/io/images.py` metadata contract:
  - shape
  - dtype
  - spacing
  - affine SHA256
  - format
- `evaluate_predictions` now fails closed on prediction/label shape mismatch.
- NIfTI prediction/label spacing and affine mismatches are fatal by default.
- Evaluation outputs now include geometry metadata per case.
- HD95/AVD now receive label spacing when available.
- Added shared `normalize_nonzero_channelwise` preprocessing policy and reused it for train-time MONAI transforms and standalone inference normalization.
- Added `scripts/verify_wmh2017_download_evidence.py`.
- Packaged non-raw official WMH2017 download evidence under `evidence/wmh2017_download_2026-06-16/`.
- Updated source/finding/review registries to reflect evidence captured but human review still open.
- Improved `verify_release_package.py` so repo-external `--out` fails with an actionable error instead of an unhandled relative-path exception.

### Tests added
- preprocessing train/inference parity unit tests
- prediction/label shape mismatch evaluation test
- geometry metadata recording test
- release verifier repo-boundary test
- download evidence verifier pass/fail/raw-image rejection tests

### Boundary
No raw WMH2017 NIfTI data is packaged. No real training/evaluation result is claimed. Source/license, official evaluator parity, medical review, privacy/security review, Preview, and Release remain unapproved.
