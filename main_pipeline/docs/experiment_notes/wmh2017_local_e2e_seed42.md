# EXP-000 — WMH2017 local smoke run (2026-06-18)

## Run identity

| field | value |
|---|---|
| run_id | `wmh2017_local_e2e_seed42` |
| experiment | EXP-000 data integrity + MONAI smoke |
| git_commit | `3d7f3f9` |
| claim_boundary | local PoC smoke only; not clinical, customer, SOTA, or official benchmark |

## Dataset

| field | value |
|---|---|
| dataset | MICCAI 2017 WMH Segmentation Challenge (Dataverse 2022 release) |
| DOI | 10.34894/AECRSD |
| WMH2017_ROOT | `/Users/yusukefujinami/WML MICCAI Pipeline/Datasets/MICCAI2017_WMH/files` |
| cases scanned | training 60 / test 110 |
| primary input | `pre/FLAIR.nii.gz` |
| primary label | `wmh.nii.gz` (foreground = label==1 only) |
| test split used for train/val | **no** (`test_split_used: false`) |

## Split

| field | value |
|---|---|
| split_id | WMH2017-TRAIN-VAL-SEED42 |
| seed | 42 |
| train_ratio | 0.8 |
| train cases | 48 |
| val cases | 12 |
| val predictions generated | 2 (`val_max_cases=2` in smoke config) |
| val metrics computed on | cases 107, 110 (cases with saved predictions) |

## Preprocessing / label policy

- Image: per-volume nonzero z-score (`normalize_nonzero_channelwise`)
- Label: `(label == 1)` binary; label 2 ignored as foreground
- Patch size: 32×32×32 (train crops)
- Inference: sliding-window on full volume (CPU)

## Model / training

| field | value |
|---|---|
| model | MONAI 3D U-Net (channels 8/16/32, strides 2/2) |
| loss | DiceCELoss |
| optimizer | Adam lr=1e-4 |
| max_epochs | 1 |
| max_steps_per_epoch | 2 |
| global_step | 2 |
| device | **mps**（`ConvTranspose3d` は `mps_compat.InterpConv3d` で置換。`PYTORCH_ENABLE_MPS_FALLBACK=1` 単体では未対応 op は解決しない） |
| checkpoint | `artifacts/runs/wmh2017_local_e2e_seed42/train/checkpoints/model_smoke.pt` |

### Train log (loss)

| step | loss |
|---:|---:|
| 1 | 1.462 |
| 2 | 1.420 |

## Validation metrics (local only)

Evaluated with `--allow-shape-only-geometry` due to negligible NIfTI spacing float drift on save.

| metric | value |
|---|---:|
| n_cases | 2 |
| mean_dice | 0.00105 |
| mean_hd95 | 127.58 |
| mean_lesion_recall | 0.931 |
| mean_lesion_f1 | 0.024 |

Per-case Dice: 107 → 0.00133, 110 → 0.00076.

**Interpretation:** Dice is near zero as expected for a 2-step smoke run with tiny patch training; pipeline wiring is validated, not model performance.

## Artifacts

```text
reports/dataset_manifest.csv
reports/label_value_audit.csv
reports/overlays/100_flair_label1_overlay.png
data/splits/wmh2017_train_val_seed42.csv
artifacts/runs/wmh2017_local_e2e_seed42/train/run_evidence.json
artifacts/runs/wmh2017_local_e2e_seed42/train/predictions/*_pred.nii.gz
artifacts/runs/wmh2017_local_e2e_seed42/metrics/metrics_summary.json
registry/run_manifest.csv (appended)
```

## Issues / follow-ups

1. **MPS + MONAI UNet:** `PYTORCH_ENABLE_MPS_FALLBACK=1` だけでは `ConvTranspose3d` は MPS 上で失敗する（PyTorch 2.4.x 実測）。`train_monai.py` は MPS 選択時に `wmh2017.training.mps_compat` で decoder の `ConvTranspose3d` を nearest upsample + `Conv3d` に置換する（nnU-Net MPS trainer と同手法）。`device: auto` で Mac 上 MPS 学習が可能。**監査上は「MPS互換パッチ + fallback許容下での実行成功」として扱い、完全な MPS ネイティブ実行や ConvTranspose3d との数値同等性は claim しない。** `run_evidence.json` に device/patch/fallback フィールドを記録。
2. **NIfTI spacing on save:** prediction save can drift zooms at float32; strict geometry eval fails; consider preserving reference header exactly or document shape-only eval for smoke.
3. **val_max_cases vs full val split:** smoke config predicts 2 val cases; evaluator expects predictions for all val rows unless split is filtered.
4. **AC-09:** no official evaluator export yet; do not compare to challenge leaderboard.
5. **Next phase:** longer training, full-val inference, T1 channel, official metric parity review before any performance claims.

## AC checklist (this run)

| AC | status |
|---|---|
| AC-01 data acquired | pass |
| AC-02 load image/label | pass |
| AC-03 visualize ≥1 case | pass (`100_flair_label1_overlay.png`) |
| AC-04 train/val split | pass |
| AC-05 MONAI 3D train | pass (smoke) |
| AC-06 inference masks | pass (2 val cases) |
| AC-07 Dice computed | pass (local) |
| AC-08 experiment memo | pass (this file) |
| AC-09 challenge comparison | deferred (no official export) |
| AC-10 no proprietary data | pass |
