# Refactor implementation summary

Implemented changes:

- Added deterministic dependency pins in `requirements-lock.txt`.
- Added `configs/wmh2017_monai_smoke.yaml`.
- Replaced scaffold-only `train_monai.py` with executable MONAI smoke training.
- Replaced scaffold-only evaluation script with local validation metrics generation.
- Added `src/wmh2017/io/images.py` for NIfTI and `.npy` fixture IO.
- Added `src/wmh2017/evaluation/evaluate_predictions.py`.
- Added split artifact hashes and site/scanner summaries.
- Added release package structural verifier.
- Added minimal full pipeline shell script.
- Added GitHub Actions workflow for structural CI and security/supply-chain scans.
- Added fixture-based integration tests for evaluation and IO.
- Added engineering validation and release decision documentation.

Claim boundary:

This is still a local research PoC package. It does not contain real WMH2017 data, run logs, official challenge results, reviewer approval, monitoring, rollback rehearsal, clinical validation, or customer-release approval.
