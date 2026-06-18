
## 0.2.2 - 2026-06-18

- Added raw WMH2017 `files/` root manifest generation evidence controls.
- Added SHA256SUMS expected-hash recording without copying raw medical images.
- Added optional NIfTI geometry inspection for dataset manifest and label audit.
- Added label audit JSON summary and fail-closed mode.
- Added official evaluator parity comparison against supplied official CSV/TSV/JSON exports.
- Added end-to-end local evidence orchestration script.
- Added tests for checksum manifest handling and official parity gates.
- Release state remains `READY_FOR_REQUIREMENTS_REVIEW`; no clinical/customer/leaderboard claim is authorized.

# Changelog

All notable changes to this repository are documented in this file.

## v0.0.0.0 - 2026-06-16

### Added — initial inheritance baseline

- Repository skeleton matching the `yskfuji/*-reproducible-pipeline` portfolio convention.
- Tier S code (verbatim from `isles2022-2d3d-blend-reproducible-pipeline` @ tag v0.0.0.1):
  - `core/pipeline/src/training/losses.py`
  - `core/pipeline/src/training/utils_train.py`
  - `core/pipeline/src/models/convnext_nnunet_seg.py`
  - `core/pipeline/src/models/input_adapters.py`
  - `core/pipeline/src/preprocess/utils_io.py`
  - `core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`
  - `core/pipeline/scripts/cross_arch_ensemble_native.py`
  - `core/pipeline/tools/make_manifest.py`
- Tier A code (renamed + import-rewired + `# DEFERRED_WMH_REVIEW:` header stubs):
  - `core/pipeline/src/datasets/wmh_dataset.py` (← `isles_dataset.py`)
  - `core/pipeline/src/training/train_wmh_25d_convnext.py` (← `train_isles_25d_convnext_fpn.py`)
  - `core/pipeline/src/evaluation/evaluate_wmh_25d.py` (← `evaluate_isles_25d.py`)
  - `core/pipeline/src/evaluation/evaluate_wmh_25d_ensemble.py` (← `evaluate_isles_25d_ensemble.py`)
  - `core/pipeline/src/evaluation/metrics_segmentation.py`
- Tier B portfolio framework: `LICENSE`, `NOTICE`, `CITATION.cff`, `AUDIT_MAP.md`,
  `ROADMAP.md`, `MODEL_CARD.md`, `sample_manifest.json`, bilingual READMEs,
  bilingual entry guides under `wmh2017/`.
- Tier C methodology transfer:
  - `docs/experiment_journey.md` — lessons distilled from ISLES (median is
    misleading, heterogeneous > homogeneous, per-case oracle analysis as cheapest
    lever, MPS workaround), plus the WMH plan keyed to the kickoff brief's
    AC-01 ... AC-10.
  - `docs/inheritance/inheritance_map.md` — exact file-by-file record of what
    came verbatim, what was renamed, what was deliberately NOT copied.
- No-data smoke test patched for WMH (14-channel input = 7 offsets × FLAIR+T1).

### Excluded (intentional)

- All ISLES configs (DWI/ADC/FLAIR-specific hyperparameters).
- ISLES per-case evaluation artifacts (`artifacts/eval_runs/cross_arch_v0.0.0.1/`).
- ISLES release notes and CHANGELOG history.
- Any trained weights.

## 0.1.0-refactor-smoke — Cursor/MONAI scaffold

- Added lightweight `src/wmh2017` scaffold for WMH2017 label policy, split policy, voxel Dice, manifest scanning, and run evidence.
- Added Cursor start guide, policy docs, task plan, and agent handoff.
- Added dataset/split/train config templates.
- Added source/dataset/split/run/metric registry schemas.
- Added unit tests for label policy, split leakage, manifest schema, and metric golden cases.
- Distribution package excludes `.git`, `.DS_Store`, `__pycache__`, `*.pyc`, and `__MACOSX`.
- Release state remains `READY_FOR_REQUIREMENTS_REVIEW`.


## 0.2.0 - MLOps executable smoke refactor

- Added pinned dependency lock baseline.
- Added MONAI smoke-training implementation.
- Added local prediction evaluation implementation.
- Added NIfTI/NumPy image IO layer.
- Added run evidence and release package verification.
- Added CI workflow for unit/structural/security gates.
- Added engineering validation plan and release decision record.
- Preserved claim boundary: research PoC only; no clinical/customer/production/SOTA claim.
