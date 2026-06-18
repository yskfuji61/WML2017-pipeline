# MICCAI 2017 WMH パイプライン — エントリーガイド (JA)

このリポジトリの WMH 固有部分に関する詳細ガイド。リポジトリ全体のオリエンテーションは top-level [`README_ja.md`](../README_ja.md) を参照。

## 現状

| 状態 | 項目 |
|---|---|
| ✅ 継承済 | 汎用 2.5D ConvNeXt + UNet decoder（channel 非依存） |
| ✅ 継承済 | MPS 互換 nnU-Net 3D trainer (`nnUNetTrainer_MPS3D_500epochs`) |
| ✅ 継承済 | Cross-architecture 確率融合 + 適応閾値 |
| ✅ 継承済 | TverskyOHEMBCE / DiceFocal / Boundary loss ファミリー |
| 🚧 DEFERRED_WMH_REVIEW | `wmh_dataset.py` — FLAIR + T1 + mask パスを記入 |
| 🚧 DEFERRED_WMH_REVIEW | FLAIR + T1 用 training config（2 modality、~14 input channel） |
| 🚧 DEFERRED_WMH_REVIEW | 5-metric 評価スイート (Dice + HD95 + AVD + F1 + Recall) |
| 🚧 DEFERRED_WMH_REVIEW | Scanner-stratified split CSV (Utrecht / Singapore / GE3T 各 20 ケース) |
| 🚧 DEFERRED_WMH_REVIEW | "Other pathology" mask 除外 (class 2 → 無視) |

## データセット情報 (MICCAI 2017 WMH)

- 提供元: https://wmh.isi.uu.nl/
- 学習セット: 60 ケース × 3 scanner（Utrecht / Singapore / GE3T 各 20）
- テストセット: 110 ケース（運営側 hold-out、grand-challenge.org に submit）
- Modality: FLAIR (primary), T1 (FLAIR に位置合わせ済)
- Mask: voxel-wise 2 値；class 1 = WMH、class 2 = "other pathology"（評価除外）
- 文献: Kuijf et al., IEEE TMI 2019, doi:10.1109/TMI.2019.2905770
- 公式指標: Dice, HD95, AVD, F1 (lesion-wise), Recall (lesion-wise)

## 推奨フェーズ順序（kickoff brief 準拠）

### Phase 1 — Catch-up baseline (目標: 2026 年 6 月末)

1. **AC-01 / AC-03**: データダウンロード、1 ケース可視化
2. **AC-02 / AC-04**: WMH 固有 dataset loader 実装、scanner-stratified split
3. **AC-05**: **MONAI 3D U-Net** baseline 学習。kickoff brief で MONAI 標準モデル指定 — 継承された 2.5D ConvNeXt は Phase 2 用
4. **AC-06 / AC-07**: 予測 + Dice 計算、その後 5-metric MICCAI スイートに拡張
5. **AC-08 / AC-09 / AC-10**: 実験ノート、公開 WMH チャレンジ結果との比較、独自データ / クラウド未使用の確認

### Phase 2 — ISLES recipe を適用

Phase 1 で baseline が確立してから:

1. 継承された 8-model 2.5D ConvNeXt アンサンブル（4 config × 2 seed）を WMH 用 hyperparam で学習（より低い `pos_slice_weight`、より狭い `slice_offsets`、`in_channels=14`）
2. `nnUNetv2_train ... -tr nnUNetTrainer ...` で 3-fold 2D nnU-Net 学習
3. `nnUNetv2_train ... -tr nnUNetTrainer_MPS3D_500epochs ...` で 2-fold MPS 3D nnU-Net 学習
4. パラメータ再校正済み `cross_arch_ensemble_native.py` を実行

### Phase 3 — Per-case oracle 分析（最大効率の lever）

ISLES の知見: 1 時間の per-case oracle threshold 分析で +0.014 mean Dice を獲得 — ~460 時間の追加 model 訓練を上回る。WMH 計画:

1. 閾値 [0.05, 0.10, ..., 0.95] でケース毎 Dice を計算
2. 「drag」ケース（最低 Dice）を特定し、系統的 over/under-prediction の方向を確認
3. Oracle を模倣するヒューリスティック構築（volume-conditioned threshold、scanner-conditioned threshold、lesion-load-conditioned threshold など）
4. 簡単なケースで heuristic が誤発火しないことを検証

## 継承モデルの読み込み方

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

First conv は ImageNet 3-ch pretrained 重みから 14-ch に `src.models.input_adapters.adapt_first_conv` 経由で自動適応。

## MPS 互換 3D nnU-Net trainer の使い方

```bash
# 1. trainer ファイルを nnunetv2 install に配置
cp core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py \
   $(python -c "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/network_architecture/

# 2. 学習 (WMH Dataset folder で nnUNetv2_plan_and_preprocess 実行後)
nnUNetv2_train <DATASET_ID> 3d_fullres 0 \
    -tr nnUNetTrainer_MPS3D_500epochs -device mps
```

trainer の monkey patch は `ConvTranspose3d` を nearest-neighbor upsample (`view`+`expand`+`reshape`、MPS native) + 3×3×3 Conv3d で置換。CPU フォールバックなし、Apple Silicon で native 動作。

## Smoke test

```bash
python scripts/smoke_test.py --use_dummy_data
```

smoke test は WMH 14-ch 入力 shape、cross-arch 適応閾値ロジック、manifest を検証。WMH 固有 UNRESOLVED_PLACEHOLDER stub が埋められたかは **検証しない**。
