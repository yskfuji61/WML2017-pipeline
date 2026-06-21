# Experiment journey — methodology carried from ISLES, plan for WMH

**Language:** English | [Japanese](experiment_journey_ja.md)

This document carries forward the lessons distilled in the ISLES 2022 sibling
repo's [`experiment_journey.md`](https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline/blob/main/docs/experiment_journey.md)
and translates them into a concrete plan for the MICCAI 2017 WMH segmentation
catch-up phase.

## Section A — Lessons inherited from ISLES (mandatory reading)

The ISLES pipeline reached **mean Dice 0.7527** after ≈ 460 h of MPS training
compute and ≈ 1 h of post-processing analysis. The honest cost accounting:

| Investment | Δ test Dice |
|---|---:|
| Knowledge distillation experiments (3 variants, ~30 h) | -0.007 (failed) |
| ConvNeXt-V2-Base + UperNet (2 configs, ~60 h) | -0.012 to -0.020 (failed) |
| 2D nnU-Net fold-3 retrain (~80 h) | -0.007 (no improvement) |
| 3D nnU-Net f0+f1 (~120 h) | **+0.0035** (heterogeneous 2D+3D blend) |
| 3D nnU-Net f2, f3, f4 added later (~170 h) | 0 to negative |
| **Subtotal model training** | **~460 h → +0.0035** |
| Per-case oracle threshold analysis (~1 h) | **+0.0140 (adaptive thresholding)** |

### Four lessons that apply directly to WMH

1. **Median dice is a distractor.** ISLES had median 0.83 and mean 0.74 — the
   median hides the cases that drag your benchmark. For WMH always report
   **mean** Dice (and the other 4 MICCAI metrics).

2. **Heterogeneous architecture diversity > homogeneous fold count.** Adding a
   4th and 5th fold of the *same* architecture plateaued or hurt. Adding a
   *different* architecture (3D nnU-Net to 2D nnU-Net + 2.5D ConvNeXt) was
   the breakthrough. WMH plan: don't grind out 5 folds of the same arch
   before trying a second architecture.

3. **Per-case oracle threshold analysis is cheap and often the real lever.**
   Compute the threshold that maximizes Dice per case (cheats with GT), then
   look for a heuristic that mimics it. In ISLES, large-prediction cases
   needed lower threshold (0.30 → 0.03) — exactly +0.014 mean Dice from a
   ~1 hour script.

4. **MPS limitations are workable.** `ConvTranspose3d` and `F.interpolate(mode='nearest'|'trilinear')`
   are unimplemented on Apple MPS, but `view`+`expand`+`reshape` provides a
   mathematically exact nearest-neighbor upsample that runs natively. See
   `core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`.

## Section B — Inversions to expect on WMH (calibration warnings)

Naïvely applying the ISLES recipe to WMH will likely misfire on these axes;
plan to **recalibrate** them rather than copy-paste defaults.

### B.1 Lesion density / positive-slice weight

ISLES DWI lesions are sparse (often < 0.5% of brain voxels). `pos_slice_weight=50`
made sense. WMH lesions are typically denser per slice (4–15% of brain voxels
for moderate-load patients). Start with `pos_slice_weight=5–10` and adjust based
on training-time positive sample ratio.

### B.2 Adaptive threshold direction

ISLES finding: **large** lesions are under-predicted at base threshold 0.30
→ switch DOWN to 0.03 when predicted volume > 4000 voxels.

WMH model behavior is empirically the opposite for many recipes: models can
over-predict periventricular halos (false-positive WMH near the lateral
ventricles). The adaptive heuristic should be re-derived from WMH per-case
oracle analysis — it may end up as "switch UP when predicted volume is large"
or it may need a scanner-conditioned threshold instead.

### B.3 Modalities

ISLES used 3 modalities (DWI / ADC / FLAIR), `in_channels = 21` (7 offsets × 3).
WMH uses 2 modalities (FLAIR / T1), default `in_channels = 14` (7 offsets × 2).
The inherited `ConvNeXtNnUNetSeg` handles arbitrary input channels via
`input_adapters.adapt_first_conv` — no code change, only `in_channels` arg.

### B.4 Slice thickness / offsets

ISLES DWI native spacing ≈ 2 mm. Offsets `[-5, -3, -1, 0, 1, 3, 5]` span 10 mm.
WMH FLAIR is often thicker (3 mm or anisotropic). Same offset list would span
15 mm — possibly too wide. Re-tune `slice_offsets` after measuring WMH spacing
distribution.

### B.5 Evaluation metric set

ISLES used mean Dice only. **WMH uses 5 metrics** (Kuijf et al., 2019):
- Dice
- HD95 (95-th percentile Hausdorff distance, mm)
- AVD (Absolute Volume Difference, %)
- F1 (lesion-wise detection)
- Recall (lesion-wise recall)

Extend `evaluate_wmh_25d_ensemble.py` to report all five.

## Section C — Plan for the catch-up phase (per kickoff brief)

The kickoff brief defines a clear acceptance-criteria-driven catch-up phase
(AC-01 through AC-10). The plan below maps each AC to inherited assets and to
the WMH-specific work that has to fill the UNRESOLVED_PLACEHOLDER stubs.

### Phase 1: data layer (AC-01 to AC-04)

| Task | Uses inherited | WMH-specific work |
|---|---|---|
| AC-01: Obtain WMH challenge data | n/a | Download from https://wmh.isi.uu.nl/, accept DUA |
| AC-02: Load images / labels | `wmh_dataset.py` skeleton | Fill `DEFERRED_WMH_REVIEW:` for FLAIR + T1 + mask paths, exclude class 2 |
| AC-03: Visualize FLAIR + mask | `nibabel` / `napari` ad hoc | none (just verify) |
| AC-04: Train/val split | `pandas` in `wmh_dataset.py` | Build scanner-stratified CSV (Utrecht / Singapore / GE3T 20 cases each) |

### Phase 2: baseline training (AC-05, AC-06)

Per kickoff brief, **start with MONAI / PyTorch standard 3D segmentation model**,
not the inherited heterogeneous ensemble. Reasons:
- AC-05 wants a baseline, not SotA;
- MONAI 3D U-Net is well-documented and easy to defend to medical reviewers;
- the inherited ensemble is "Phase 2 / SotA push" work (Phase 2 in ROADMAP).

For Phase 1 baseline:
```python
from monai.networks.nets import UNet
model = UNet(spatial_dims=3, in_channels=2, out_channels=2,  # FLAIR+T1, BG+WMH
             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
```

### Phase 3: evaluation (AC-07)

Start with Dice (AC-07 minimum), then add the 5-metric MICCAI suite. Use
`evaluate_wmh_25d_ensemble.py` as a starting point — the structure handles
loading probs, applying threshold + min_size, and computing Dice; HD95/AVD/F1
need to be added.

### Phase 4: reality check (AC-09)

Compare to published WMH challenge scores:
- 2017 challenge winner (Sysu_Med, Li et al.): Dice ≈ 0.80
- Post-challenge nnU-Net baseline: Dice ≈ 0.82–0.85
- Modern multi-arch ensembles: Dice ≈ 0.85–0.87

If our baseline is < 0.70 Dice we should *not* assume the inherited recipe is
the issue — first check:
1. Are we excluding class 2 ("other pathology")? — common mistake.
2. Are we evaluating per-subject (macro) or per-voxel (micro)? — challenge uses macro.
3. Are we using both modalities (FLAIR + T1)? — T1 contributes ~+0.02 Dice.
4. Are we honouring scanner stratification? — un-stratified can bias eval.

These four checks have caught most "low Dice on first attempt" failures in
the literature. Address them before reaching for the inherited adaptive-threshold
machinery.

## Section D — Stop conditions / human review gates

Per the kickoff brief, these stop conditions apply:
1. Data turns out to contain PII / patient identifiers → halt, escalate.
2. Cloud upload becomes necessary → halt, escalate.
3. Dice is anomalously low and the cause isn't clear → halt, document, ask.
4. Data format or label semantics are unclear → halt, ask.
5. Proprietary data ingestion is contemplated → halt, escalate.

Human review gates (from kickoff brief):
1. First successful training run on public data.
2. First Dice number reported.
3. First comparison to published challenge scores.
4. Any move to proprietary data.
5. Any move to cloud compute.
6. Any draft of customer-facing output.
7. Any wording of medical claims.
8. End-of-June final review before the July 2026 production phase.

## Section E — References

- ISLES 2022 sibling repo experiment journey (this doc's seed):
  https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline/blob/main/docs/experiment_journey.md
- MICCAI 2017 WMH Segmentation Challenge: https://wmh.isi.uu.nl/
- Kuijf et al., "Standardized Assessment of Automatic Segmentation of White
  Matter Hyperintensities and Results of the WMH Segmentation Challenge,"
  IEEE Transactions on Medical Imaging, 2019. doi:10.1109/TMI.2019.2905770

## Section F — Catch-up phase results (June 2026, local CV)

These are local validation results only. They are NOT official-benchmark,
clinical, customer, production, or SOTA claims. The published challenge scores in
Section C remain external reference values, not comparisons we are making here.

### F.1 5-fold cross-validation (honest, variance-aware)

After the Phase A governance refactor (ADR-0007) and the Phase B1 k-fold
foundation, a 5-fold site-stratified CV was run with the A2 recipe
(TverskyFocal, alpha=0.3/beta=0.7/gamma=1.33) + cosine LR, 100 epochs/fold,
seed 42. Summary: `reports/cv/cv_summary_a2cv_cosine_seed42.json`.

| metric | CV mean +/- std (n=5) | per-fold |
|---|---|---|
| mean_dice | 0.614 +/- 0.037 | 0.660 / 0.629 / 0.619 / 0.560 / 0.601 |
| mean_lesion_recall | 0.207 +/- 0.038 | 0.173 / 0.212 / 0.177 / 0.205 / 0.268 |
| mean_lesion_f1 | 0.297 +/- 0.047 | - |

Gate judgment: Phase A gate (Dice 0.65 / Recall 0.35) NOT met; Phase B gate
(Dice 0.72) NOT met.

### F.2 The single-split measurement was optimistic

A prior single-split A2-100ep run reported Dice 0.645. The honest 5-fold CV mean
is 0.614 +/- 0.037 with fold range 0.560-0.660. This confirms lesson #1 from
Section A in a new way: a single validation split over-stated performance, and
the variance across folds (~0.10 spread) is large relative to the gaps we are
chasing. CV is now the unit of measurement for any performance claim.

### F.3 Next levers (in priority order)

1. Recall is the binding constraint (0.207 vs 0.35 target). The cheapest
   config-only lever is the loss FN weight (Tversky `beta`) plus light positive
   sampling (`training.sampling.pos`), evaluated on the CV foundation. Note the
   earlier A3 attempt (pos=3 alone) lowered Dice to 0.562, so positive sampling
   must be paired with the FN-weighted loss and validated for Dice regression.
2. Per-case oracle threshold analysis (Section A lesson #3) — cheap, may recover
   recall/Dice without retraining.
3. Heterogeneous-architecture ensemble (Section A lesson #2: hetero-arch beats
   homogeneous fold count) — the ConvNeXt 2.5D path is the second architecture.
