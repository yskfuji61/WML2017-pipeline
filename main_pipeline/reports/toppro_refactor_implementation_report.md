# TopPro Refactor Implementation Report

## Verdict

Implementation completed for the P0/P1 audit fixes that can be safely applied
without access to the raw WMH2017 `files/` root and without changing the project
scope.

Current state remains:

```text
READY_FOR_REQUIREMENTS_REVIEW
NOT_READY_FOR_PREVIEW
NOT_READY_FOR_RELEASE
```

## Implemented changes

- Added image geometry metadata contract.
- Added fail-closed prediction/label shape validation.
- Added default NIfTI spacing and affine equality checks.
- Added case-level geometry metadata to evaluation outputs.
- Passed label spacing to HD95/AVD metrics when available.
- Added shared preprocessing policy for train/inference normalization parity.
- Added official-download evidence package without raw NIfTI images.
- Added evidence verifier for download metadata/checksum logs.
- Updated source/finding/review registries to reflect evidence captured but
  human review still open.
- Improved release verifier repo-boundary error handling.
- Added regression/unit tests for all modified behavior.

## Verification performed in this environment

```text
python -m compileall scripts src tests
status: passed

python -m pytest -q
status: passed
tests: 32 passed

python scripts/verify_wmh2017_download_evidence.py --evidence-dir evidence/wmh2017_download_2026-06-16 --out reports/wmh2017_download_evidence_verification.json
status: passed
downloaded_file_count: 1791
sha256_entry_count: 1791
sha256_verified_ok_count: 1791
raw_medical_files_in_evidence_package: 0

python scripts/verify_release_package.py --repo-root . --out reports/full_package_manifest.json --package-version 0.2.1 --package-id WMH2017-LOCAL-POC-SCAFFOLD-0.2.1
status: passed
```

## Verification not performed

- Real WMH2017 dataset manifest generation from the raw local `files/` root.
- Real label audit against NIfTI masks.
- Real MONAI training.
- Real prediction/evaluation on validation cases.
- Official evaluator parity.
- Ruff/Bandit/pip-audit/detect-secrets execution in this environment.
- Human source/license, medical, security/privacy, and release review.

## Residual blockers

- Source/license review is not approved.
- Official evaluator code/hash/parity is not captured.
- Real run evidence is absent.
- Review/approval records remain `NOT_APPROVED`.
- Clinical/customer/commercial/cloud/proprietary-data use remains blocked.
