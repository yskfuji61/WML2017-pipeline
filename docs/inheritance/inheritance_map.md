# Inheritance map — what came from `isles2022-2d3d-blend-reproducible-pipeline`

This repository was bootstrapped from the sibling repo
`isles2022-2d3d-blend-reproducible-pipeline` (commit baseline: `925ae8c`, tag
`v0.0.0.1`). Following table records, for every inherited file, whether it was
copied **verbatim**, **renamed**, or **renamed-with-stubs**.

## Tier S — verbatim (no edits)

These files are dataset-agnostic infrastructure. Identical bytes vs. source.

| File in wmh2017 | Source in isles2022-2d3d-blend |
|---|---|
| `core/pipeline/src/training/losses.py` | same path |
| `core/pipeline/src/training/utils_train.py` | same path |
| `core/pipeline/src/models/convnext_nnunet_seg.py` | same path |
| `core/pipeline/src/models/input_adapters.py` | same path |
| `core/pipeline/src/preprocess/utils_io.py` | same path |
| `core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py` | same path |
| `core/pipeline/scripts/cross_arch_ensemble_native.py` | same path |
| `core/pipeline/tools/make_manifest.py` | same path |
| `LICENSE` | same path (Apache 2.0) |
| `.gitignore` | same path |

## Tier A — renamed + import rewires + DEFERRED_WMH_REVIEW header stubs

These files retain ~100% of the source logic but rename ISLES tokens and add
explicit DEFERRED_WMH_REVIEW markers at the top describing what needs WMH-specific
calibration before they actually do anything useful.

| File in wmh2017 | Source in isles2022-2d3d-blend | Edits |
|---|---|---|
| `core/pipeline/src/datasets/wmh_dataset.py` | `core/pipeline/src/datasets/isles_dataset.py` | rename + DEFERRED_WMH_REVIEW header listing: scanner-stratified CSV, FLAIR+T1 channels, "other pathology" mask handling |
| `core/pipeline/src/training/train_wmh_25d_convnext.py` | `core/pipeline/src/training/train_isles_25d_convnext_fpn.py` | import path → `..datasets.wmh_dataset`; DEFERRED_WMH_REVIEW header listing: slice_offsets, pos_slice_weight, in_channels |
| `core/pipeline/src/evaluation/evaluate_wmh_25d.py` | `core/pipeline/src/evaluation/evaluate_isles_25d.py` | import rewire only |
| `core/pipeline/src/evaluation/evaluate_wmh_25d_ensemble.py` | `core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py` | import rewire + DEFERRED_WMH_REVIEW header listing: extend to 5-metric MICCAI suite, exclude "other pathology" voxels |
| `core/pipeline/src/evaluation/metrics_segmentation.py` | same path | verbatim (kept under Tier A grouping for evaluation cohesion) |
| `scripts/smoke_test.py` | same path | docstring + default channel count adjusted from 21 (ISLES DWI+ADC+FLAIR×7) to 14 (WMH FLAIR+T1×7) |

## Tier B — portfolio framework (template adapted)

These are repo-level meta documents recreated for WMH from the ISLES templates.

| File in wmh2017 | Source template | Status |
|---|---|---|
| `NOTICE` | ISLES NOTICE | WMH-specific text (cites WMH challenge, links wmh.isi.uu.nl) |
| `CITATION.cff` | ISLES CITATION | new metadata (`wmh2017-reproducible-pipeline`, references ISLES repo + Kuijf et al. 2019) |
| `AUDIT_MAP.md` | ISLES AUDIT_MAP | WMH-specific reading order + audit highlights + recipe sketch |
| `ROADMAP.md` | ISLES ROADMAP | WMH-specific 4-phase plan (Phase 0 inheritance done, Phase 1 catch-up next) |
| `README.md` / `README_ja.md` | ISLES READMEs | WMH-tailored headlines |
| `MODEL_CARD.md` | ISLES MODEL_CARD | WMH-specific (template — to be filled when first WMH models train) |
| `sample_manifest.json` | ISLES sample_manifest | regenerated for WMH file set |

## Tier C — methodology (knowledge transfer, not code)

| File in wmh2017 | What it captures |
|---|---|
| `docs/experiment_journey.md` (+ `_ja.md`) | The four lessons from ISLES that should shape WMH work:<br>1. Median is misleading — report mean Dice on the official metric set.<br>2. Heterogeneous arch addition beats homogeneous fold addition.<br>3. Per-case oracle threshold analysis is cheap (~1 h) and often beats months of training.<br>4. MPS 3D ConvTranspose3d workaround via `view+expand+reshape` nearest-neighbor upsample. |

## Things deliberately NOT inherited

These were present in `isles2022-2d3d-blend-reproducible-pipeline` v0.0.0.1 but
intentionally excluded from this WMH inheritance:

- `core/pipeline/configs/train_convnext_v{2,2_aug,3_aug,3_dilated}_1mm.yaml`
  — hyperparameters tuned for ISLES DWI/ADC/FLAIR at 2mm spacing. WMH needs
    new configs after data inspection (Phase 1).
- `artifacts/eval_runs/cross_arch_v0.0.0.1/` — ISLES 25-case `case_id`s. Not
  applicable; WMH will have its own per-case evidence artifact directory.
- `docs/releases/v0.0.0.1*.md` — ISLES release history.
- `CHANGELOG.md` — fresh, WMH-versioned changelog will accompany v0.0.0.0.
- `docs/weights.md` — ISLES tarball SHA-256 hashes. WMH will get its own when
  weights exist.
- `data/splits/*.csv` (ISLES split CSV references).

## Verification

After inheritance, the no-data smoke test still passes structurally:

```bash
python scripts/smoke_test.py --use_dummy_data
```

This confirms:
- the inherited `ConvNeXtNnUNetSeg` accepts the WMH-defaulted 14-channel
  input (7 offsets × 2 modalities for FLAIR + T1) and produces the expected
  `(B, 1, 256, 256)` logit tensor shape;
- the inherited `cross_arch_ensemble_native.py` adaptive-threshold logic
  fires correctly on synthetic large-volume cases and not on small ones.

What it does **not** verify:
- WMH-specific dataset I/O, since `wmh_dataset.py` still has DEFERRED_WMH_REVIEW stubs.
- WMH-tuned hyperparameters (no WMH configs yet).
- WMH-aware metrics (HD95, AVD, lesion F1, Recall) — current evaluator reports
  Dice only.
