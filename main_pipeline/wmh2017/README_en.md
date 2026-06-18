# MICCAI 2017 WMH Pipeline — entry guide (EN)

This is the in-depth entry doc for the WMH-specific portion of this
repository. For repo-level orientation see the top-level
[`README.md`](../README.md).

## What's here vs what's coming

| Status | Item |
|---|---|
| ✅ Inherited | Generic 2.5D ConvNeXt + UNet decoder (channel-agnostic) |
| ✅ Inherited | MPS-compatible nnU-Net 3D trainer (`nnUNetTrainer_MPS3D_500epochs`) |
| ✅ Inherited | Cross-architecture probability fusion + adaptive thresholding |
| ✅ Inherited | TverskyOHEMBCE / DiceFocal / Boundary loss family |
| 🚧 DEFERRED_WMH_REVIEW | `wmh_dataset.py` — fill in FLAIR + T1 + mask paths |
| 🚧 DEFERRED_WMH_REVIEW | Training configs for FLAIR + T1 (2 modalities, ~14 input channels) |
| 🚧 DEFERRED_WMH_REVIEW | 5-metric evaluation suite (Dice + HD95 + AVD + F1 + Recall) |
| 🚧 DEFERRED_WMH_REVIEW | Scanner-stratified split CSV (Utrecht / Singapore / GE3T, 20 each) |
| 🚧 DEFERRED_WMH_REVIEW | "Other pathology" mask exclusion (class 2 → ignored) |

## Dataset facts (MICCAI 2017 WMH)

- Source: https://wmh.isi.uu.nl/
- Train set: 60 cases × 3 scanners (Utrecht / Singapore / GE3T, 20 each)
- Test set: 110 cases held by the organizers (results submitted to grand-challenge.org)
- Modalities: FLAIR (primary), T1 (registered to FLAIR)
- Mask: voxel-wise binary; class 1 = WMH, class 2 = "other pathology" (excluded from eval)
- Reference paper: Kuijf et al., IEEE TMI 2019, doi:10.1109/TMI.2019.2905770
- Official metrics: Dice, HD95, AVD, F1 (lesion-wise), Recall (lesion-wise)

## Suggested phase order (matches kickoff brief)

### Phase 1 — Catch-up baseline (target: end of June 2026)

1. **AC-01 / AC-03**: download data, visualize one case.
2. **AC-02 / AC-04**: implement WMH-specific dataset loader and scanner-stratified split.
3. **AC-05**: train a **MONAI 3D U-Net** baseline. The kickoff brief explicitly
   asks for a MONAI standard model first — the inherited 2.5D ConvNeXt is a
   Phase 2 lever.
4. **AC-06 / AC-07**: predict + compute Dice; then extend to the 5-metric MICCAI suite.
5. **AC-08 / AC-09 / AC-10**: lab notebook, compare to published WMH challenge scores,
   confirm no proprietary data / cloud usage.

### Phase 2 — Apply ISLES recipe

After Phase 1 establishes a credible baseline:

1. Train the inherited 8-model 2.5D ConvNeXt ensemble (4 configs × 2 seeds) with
   WMH-tuned hyperparameters (lower `pos_slice_weight`, narrower `slice_offsets`,
   `in_channels=14`).
2. Train 3-fold 2D nnU-Net via `nnUNetv2_train ... -tr nnUNetTrainer ...`.
3. Train 2-fold MPS-compatible 3D nnU-Net via
   `nnUNetv2_train ... -tr nnUNetTrainer_MPS3D_500epochs ...`.
4. Run `cross_arch_ensemble_native.py` with recalibrated parameters.

### Phase 3 — Per-case oracle analysis (the cheap lever)

ISLES finding: a 1-hour per-case oracle threshold analysis bought +0.014
mean Dice — more than ~460 h of additional model training. WMH plan:
1. Compute per-case Dice at thresholds [0.05, 0.10, ..., 0.95].
2. Identify the "drag" cases (lowest Dice) and the systematic over/under-prediction direction.
3. Build a heuristic that mimics the oracle (e.g., volume-conditioned threshold,
   scanner-conditioned threshold, lesion-load-conditioned threshold).
4. Validate the heuristic does not over-fire on the easy cases.

## How to load the inherited model

```python
import torch
from src.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg

# WMH default: 7 offsets × 2 modalities = 14 channels
model = ConvNeXtNnUNetSeg(
    in_channels=14,
    backbone="convnext_tiny",
    pretrained=True,
    dec_ch=256,
    deep_sup=False,
)
model.eval()
```

The first conv is automatically adapted from the 3-channel ImageNet pretrained
weights to 14 channels via `src.models.input_adapters.adapt_first_conv`.

## How to use the MPS-compatible 3D nnU-Net trainer

```bash
# 1. Drop the trainer file into the nnunetv2 install
cp core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py \
   $(python -c "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/network_architecture/

# 2. Train (after running nnUNetv2_plan_and_preprocess on a WMH Dataset folder)
nnUNetv2_train <DATASET_ID> 3d_fullres 0 \
    -tr nnUNetTrainer_MPS3D_500epochs -device mps
```

The trainer's monkey patch replaces `ConvTranspose3d` with nearest-neighbor
upsample (via `view`+`expand`+`reshape`, MPS-native) + 3×3×3 Conv3d. No CPU
fallback — runs natively on Apple Silicon.

## Smoke test

```bash
python scripts/smoke_test.py --use_dummy_data
```

The smoke test verifies the WMH 14-channel input shape, the cross-arch
adaptive-threshold logic, and the manifest. It does **not** verify that
WMH-specific UNRESOLVED_PLACEHOLDER stubs have been filled in.
