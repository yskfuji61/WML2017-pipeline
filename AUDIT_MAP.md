# WMH2017 Pipeline — Audit Map

## Current authoritative scope

This repository is a local public-data PoC scaffold for the June 2026 WMH2017
catch-up phase. The primary route is the MONAI smoke path under `src/wmh2017/**`,
`scripts/**`, and `configs/wmh2017_monai_smoke.yaml`.

`core/pipeline/**` is inherited reference material and must not be treated as the
authoritative path for the first WMH2017 smoke run unless a later review record
promotes a specific file.

Controlled release state: `NOT_READY_FOR_PREVIEW`.

Required audit files before promotion:

```text
reports/full_package_manifest.json
registry/finding_register_wmh2017.csv
registry/review_approval_register_wmh2017.csv
registry/decision_record_register_wmh2017.csv
docs/final_evidence_binder_index.md
docs/release_state_crosswalk.md
```


This repository inherits the heterogeneous-ensemble pipeline from
`isles2022-2d3d-blend-reproducible-pipeline` and re-targets it to the MICCAI 2017
WMH Segmentation Challenge. WMH-specific code adaptations are marked with
`# DEFERRED_WMH_REVIEW:` stubs; until those are filled in, the pipeline only **structurally**
covers the WMH task — actual WMH-tuned training is the explicit next step.

## 1. Reading order

1. `./wmh2017/README.md` (Japanese) or `./wmh2017/README_en.md`
2. `./docs/inheritance/inheritance_map.md`
   — what was copied verbatim, renamed, or replaced relative to the ISLES repo
3. `./docs/experiment_journey.md`
   — methodology notes carried over from ISLES (median is misleading,
     heterogeneous > homogeneous, adaptive threshold framework) plus the
     WMH-specific plan for the catch-up phase
4. `./core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`
   — MPS-safe 3D nnU-Net trainer (drop into the nnunetv2 install location)
5. `./core/pipeline/src/datasets/wmh_dataset.py` (UNRESOLVED_PLACEHOLDER-stubbed)
6. `./core/pipeline/src/training/train_wmh_25d_convnext.py` (UNRESOLVED_PLACEHOLDER-stubbed)
7. `./core/pipeline/src/evaluation/evaluate_wmh_25d_ensemble.py` (UNRESOLVED_PLACEHOLDER-stubbed)
8. `./core/pipeline/scripts/cross_arch_ensemble_native.py`
   — cross-arch fusion + adaptive threshold (reusable as-is once WMH probs exist)

## 2. Audit highlights

- **Inheritance is intentional**, not copy-paste polish. The ISLES repo's
  TverskyOHEMBCE loss, 2.5D ConvNeXt + UNet decoder, MPS 3D ConvTranspose3d
  workaround, and adaptive-threshold post-processing are all directly applicable
  to WMH (FLAIR + T1, small multifocal lesions, MPS hardware).
- **Methodological transfer**: the ISLES experiment journey concluded that
  ~460 h of additional training compute gave +0.0035, while ~1 h of per-case
  oracle threshold analysis gave +0.0140. The WMH plan should front-load the
  oracle-threshold analysis (cheap) before committing to multi-day training.
- **Inversions to expect on WMH**:
  - **lesion density** higher → reduce `pos_slice_weight` from ISLES default 50
    to ~5–10 (over-oversampling on a dense-positive dataset hurts).
  - **adaptive threshold** ISLES heuristic was "switch low when prediction is
    large (under-segmentation)". WMH models often over-predict periventricular
    halos → may need the **opposite** direction (switch high when prediction
    is large). Re-calibrate from per-case oracle analysis.
  - **evaluation** requires 5 metrics, not just Dice (Dice + HD95 + AVD + lesion
    F1 + Recall per Kuijf et al., 2019).

## 3. Recipe sketch (to be calibrated per WMH oracle analysis)

```text
nnUNet probs = α × 2D-3fold-avg + (1-α) × 3D-2fold-avg     # α to be tuned
combined     = w × ConvNeXt-avg + (1-w) × nnUNet probs       # w to be tuned
rescue       = alpha, case_gate_agree, cc_min_overlap        # all WMH-tunable
post-process = base_thr; adaptive switch direction TBD
```

ISLES used `(α=0.6, w=0.20, rescue α=0.7, cga=0.4, cco=20, base=0.30, adaptive 0.30→0.03 when pred>4000 vox)`.
These are **starting points**, not WMH-defaults.

## 4. Excluded artifacts

- `Datasets/` — obtain MICCAI 2017 WMH data from https://wmh.isi.uu.nl/
- `runs/` and `logs/` — training/eval traces are large and reproducible from configs
- Trained weights (`*.pt`, `*.pth`) — not bundled; train from scratch
