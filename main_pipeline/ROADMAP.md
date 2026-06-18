# wmh2017-reproducible-pipeline — Roadmap

## Phase 0 — Inheritance (this release, v0.0.0.0)

- [x] Repo skeleton with portfolio meta (LICENSE / NOTICE / CITATION / AUDIT_MAP)
- [x] Inherited Tier S (generic infrastructure: losses, utils, input adapters,
      ConvNeXt model, MPS 3D nnU-Net trainer, cross-arch ensemble script,
      smoke test, make_manifest)
- [x] Inherited Tier A (dataset/training/evaluation with `# DEFERRED_WMH_REVIEW:` markers)
- [x] Methodology transfer: experiment journey docs explaining what lessons
      from ISLES apply to WMH
- [x] Inheritance map documenting what came from where
- [x] No-data smoke test passes

## Phase 1 — WMH catch-up (target: end of June 2026)

Per the kickoff brief, the 6月 catch-up phase has these acceptance criteria:

- [ ] **AC-01**: Obtain & unpack MICCAI 2017 WMH challenge data
- [ ] **AC-02**: Load images and labels (FLAIR, T1, mask) — fill `wmh_dataset.py`
      DEFERRED_WMH_REVIEW stubs
- [ ] **AC-03**: Visualize FLAIR + mask for ≥ 1 case
- [ ] **AC-04**: Build train/validation split (scanner-stratified recommended)
- [ ] **AC-05**: Train a 3D segmentation model (MONAI / PyTorch baseline)
- [ ] **AC-06**: Generate inference masks
- [ ] **AC-07**: Compute Dice (and start the 5-metric MICCAI suite: HD95, AVD,
      Lesion-F1, Recall)
- [ ] **AC-08**: Lab notebook documenting experiment conditions & results
- [ ] **AC-09**: Compare result to published WMH challenge top scores; explain
      any gap > 10%
- [ ] **AC-10**: Confirm no proprietary data / cloud / customer reports used

## Phase 2 — Inherited pipeline activation (July 2026+)

- [ ] Wire `train_wmh_25d_convnext.py` to WMH dataset (fills Tier A UNRESOLVED_PLACEHOLDER stubs)
- [ ] Train 8-model 2.5D ConvNeXt ensemble (ISLES recipe re-tuned for FLAIR+T1)
- [ ] Train nnU-Net 2D 3-fold via `nnUNetv2_train ... -tr nnUNetTrainer ...`
- [ ] Train nnU-Net 3D 2-fold via `nnUNetv2_train ... -tr nnUNetTrainer_MPS3D_500epochs ...`
- [ ] Run cross-arch ensemble with WMH-recalibrated parameters

## Phase 3 — Per-case oracle analysis & adaptive post-processing

- [ ] Per-case Dice breakdown by stage (as in
      ISLES `artifacts/eval_runs/cross_arch_v0.0.0.1/`)
- [ ] Compute oracle per-case threshold → identify systematic over/under-prediction
- [ ] Re-calibrate `cross_arch_ensemble_native.py` adaptive thresholds for WMH
- [ ] Verify the heuristic only fires on the small number of cases it should

## Phase 4 — Portfolio polish & v0.1.0 release

- [ ] Per-case evaluation evidence artifact directory
- [ ] MODEL_CARD with WMH-tuned metrics
- [ ] Bilingual experiment-journey conclusions
- [ ] Tag v0.1.0 with weights tarballs (if DUA allows GitHub Release attachment)

## Non-goals (per kickoff brief)

- ❌ Clinical diagnosis or grade classification
- ❌ Customer-facing reports
- ❌ Proprietary data ingestion
- ❌ Independent cloud-resource decisions
- ❌ Treating "high Dice" as clinically valid
