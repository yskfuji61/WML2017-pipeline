# Model Card — wmh2017-reproducible-pipeline (v0.0.0.0)

> ⚠️ **No WMH-trained checkpoints exist in this release.**
> This card describes the *inherited* architectures that the WMH pipeline will
> train against in Phase 1 / Phase 2 (see [ROADMAP.md](ROADMAP.md)). It will
> be updated with concrete WMH metrics in a future release.

## Inherited architectures (ready to train on WMH data)

| Architecture | Source | Role in WMH pipeline |
|---|---|---|
| 2.5D ConvNeXt-Tiny + UNet decoder (`ConvNeXtNnUNetSeg`) | inherited from ISLES | Phase 2 — 2.5D branch (FLAIR + T1, 7 slice offsets, 14 input channels) |
| nnU-Net v2 2D fullres (`PlainConvUNet`) | upstream nnU-Net v2 | Phase 2 — 2D nnU-Net branch (planning via `nnUNetv2_plan_and_preprocess`) |
| nnU-Net v2 3D fullres with MPS-safe up-conv | inherited (`nnUNetTrainer_MPS3D_500epochs.py`) | Phase 2 — 3D nnU-Net branch on Apple Silicon |

## Intended use

- **Primary**: scientific research and reproducibility benchmarking for the
  MICCAI 2017 WMH Segmentation Challenge.
- **Portfolio / educational**: demonstrate the inheritance-based bootstrap of
  a new medical-AI segmentation pipeline from a closely-related upstream repo
  (`isles2022-2d3d-blend-reproducible-pipeline`) and the MPS 3D ConvTranspose3d
  workaround.

## Out-of-scope use

- **NOT a medical device.** No part of this pipeline is intended for clinical
  decision making, triage, patient management, or any direct patient-facing
  application.
- **NOT validated on any patient population.** No WMH model is trained yet.
- **NOT a replacement for the ISLES sibling repo.** The ISLES recipe is for
  acute stroke DWI segmentation, not for chronic white-matter hyperintensity
  detection on FLAIR.

## Training data (planned, Phase 1)

- **Dataset**: MICCAI 2017 White Matter Hyperintensities Segmentation Challenge
- **Source**: https://wmh.isi.uu.nl/
- **Modalities**: FLAIR (primary), T1 (registered to FLAIR)
- **Train**: 60 cases × 3 scanners (Utrecht / Singapore / GE3T, 20 each)
- **Test**: 110 cases held by the organizers
- **Mask convention**: class 1 = WMH (foreground), class 2 = "other pathology"
  (must be excluded from evaluation)

## Training procedure (planned)

The catch-up phase uses a **MONAI 3D U-Net** baseline per the kickoff brief, not
the inherited heterogeneous ensemble. The ensemble path is the Phase 2 / SotA
push:
- ConvNeXt-Tiny 2.5D, 4 configs × 2 seeds = 8 models
- nnU-Net 2D 3-fold (folds 0, 1, 2)
- nnU-Net 3D 2-fold (folds 0, 1) on MPS

## Evaluation (planned)

Per MICCAI 2017 WMH Challenge spec, all 5 metrics will be reported:
- Dice
- HD95 (95-th percentile Hausdorff distance, mm)
- AVD (Absolute Volume Difference, %)
- F1 (lesion-wise detection)
- Recall (lesion-wise recall)

## Known risks before any WMH model is trained

- **Periventricular over-prediction**: FLAIR-based WMH models frequently
  produce false-positive halos near the lateral ventricles. The inherited
  adaptive-threshold heuristic was tuned for ISLES (under-prediction direction)
  and likely needs the opposite direction or an entirely different conditioning
  variable for WMH — to be re-derived from per-case oracle analysis.
- **Scanner shift**: WMH challenge subjects come from 3 scanners with notably
  different intensity distributions. Scanner-stratified split is essential to
  avoid information leak.
- **"Other pathology" class**: failing to exclude class 2 voxels from evaluation
  is a common silent bug that inflates Dice.
- **MPS memory budget**: 3D nnU-Net at patch [80, 96, 80] needs batch_size=1
  on Apple M-series; the inherited trainer already forces this.

## Weights distribution

No weights bundled in v0.0.0.0. SHA-256 hashes and download paths will appear
in `docs/weights.md` once WMH models are trained.

## Citation

See [CITATION.cff](CITATION.cff). Cite the ISLES sibling repo if the
heterogeneous-ensemble or MPS 3D workaround code is used in derivative work.

## License

Apache License 2.0 for code and this model card.
