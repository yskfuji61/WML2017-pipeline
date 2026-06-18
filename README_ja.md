# WMH2017 Pipeline — ローカル公開データPoC scaffold

> package_version: `0.2.3`
>
> local planning state: `READY_FOR_REQUIREMENTS_REVIEW`
>
> controlled release state: `NOT_READY_FOR_PREVIEW`
>
> このパッケージは、2026年6月のキャッチアップ用に、公開WMH2017データで「読み込み・分割・可視化・MONAI 3D smoke training・推論・local Dice評価」へ進むための実装/証跡scaffoldです。
>
> 実WMH2017データrun、source/license review、official metric parity、review/approval、release decisionは未完了です。臨床利用、顧客提示、独自データ処理、クラウド利用、production、SOTA/leaderboard claimは禁止です。
>
> 監査上の入口: [`README_CURSOR_START.md`](README_CURSOR_START.md)、[`docs/release_state_crosswalk.md`](docs/release_state_crosswalk.md)、[`docs/final_evidence_binder_index.md`](docs/final_evidence_binder_index.md)。

**言語:** [English](README.md) | 日本語

**MICCAI 2017 白質病変（WMH）セグメンテーションチャレンジ** (Kuijf et al., IEEE TMI 2019) のための再現可能パイプライン。姉妹リポジトリ [`isles2022-2d3d-blend-reproducible-pipeline`](https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline)（ISLES 2022 用の 2D + 3D + 2.5D 異種アンサンブル）からブートストラップし、**FLAIR + T1** 入力向けに再ターゲット。

この 0.2.3 リリースは **継承ベースライン**: コードは揃ったが、WMH 固有部分は `# DEFERRED_WMH_REVIEW:` ヘッダーでマーク。WMH データへの活性化は明示的に Phase 1 タスク（[ROADMAP.md](ROADMAP.md) 参照）。

**クイックリンク**
- エントリーガイド (EN): [wmh2017/README_en.md](wmh2017/README_en.md)
- エントリーガイド (JA): [wmh2017/README.md](wmh2017/README.md)
- **継承マップ** (何が ISLES のどこから来たか): [docs/inheritance/inheritance_map.md](docs/inheritance/inheritance_map.md)
- **実験ジャーニー** (ISLES 教訓 + WMH 計画): [docs/experiment_journey.md](docs/experiment_journey.md)
- 監査マップ: [AUDIT_MAP.md](AUDIT_MAP.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)
- 引用情報: [CITATION.cff](CITATION.cff)

## このリポジトリが今 (0.2.3) 提供するもの

- 姉妹 `yskfuji/*-reproducible-pipeline` 規約に合わせた **portfolio-grade なリポジトリスケルトン**
- **MPS 互換 nnU-Net 3D trainer**（Apple Silicon の `ConvTranspose3d` 未対応回避策）をそのまま継承 — ISLES の作業で最も再利用価値の高い単一コンポーネント
- **MONAI smoke の MPS 互換パッチ**（`src/wmh2017/training/mps_compat.py`）— MPS 選択時に `ConvTranspose3d` を nearest upsample + `Conv3d` に置換。smoke/互換性確認用であり、元構成との数値同等性は保証しない
- **cross-architecture 確率融合スクリプト** + ケース適応的閾値処理 (`core/pipeline/scripts/cross_arch_ensemble_native.py`)。WMH モデルが揃った Phase 2 で使用、それまでは task 非依存で待機
- **2.5D ConvNeXt モデル** (`ConvNeXtNnUNetSeg`) と訓練ループ、WMH キャリブレーション UNRESOLVED_PLACEHOLDER マーク付き（チャネル数、slice_offsets、pos_slice_weight）
- **データ不要 smoke test**: 継承された model と適応閾値ロジックを合成 volume で 30 秒以内検証

## このリポジトリがまだ提供しないもの

- 学習済み WMH モデル。最初の WMH モデルは Phase 1 deliverable
- WMH 用 config。ISLES 用 config は意図的に **コピーしていない**
- WMH 固有のデータ I/O。`wmh_dataset.py` は `isles_dataset.py` を rename したコピーで、FLAIR / T1 / mask パス、scanner-stratified CSV カラム、"other pathology" class 除外を DEFERRED_WMH_REVIEW マーカーで記載
- MICCAI 5-metric 評価スイート (HD95, AVD, lesion F1, Recall)。現状の evaluator は Dice のみ報告、拡張は ROADMAP Phase 1

## Phase 1 合格条件（kickoff brief より）

6月 catch-up フェーズは 10 個の合格条件あり；本 repo は AC-02 / AC-05 / AC-06 / AC-07 のスケルトンを提供。AC-01（データ取得）と AC-08 / AC-09 / AC-10（文書化とレビュー）は作業者責任。

| AC | 内容 | 継承支援 | WMH 固有作業 |
|---|---|---|---|
| AC-01 | WMH データ取得 | n/a | https://wmh.isi.uu.nl/ から、DUA 同意 |
| AC-02 | FLAIR + label 読み込み | `wmh_dataset.py` スケルトン | DEFERRED_WMH_REVIEW stub を埋める |
| AC-03 | ≥ 1 ケース可視化 | n/a (手動) | nibabel / matplotlib |
| AC-04 | train/val split | `wmh_dataset.py` の pandas | scanner-stratified CSV |
| AC-05 | baseline モデル学習 | `train_wmh_25d_convnext.py` または MONAI 3D U-Net | baseline は MONAI 使用（brief 指定） |
| AC-06 | 推論 mask | `evaluate_wmh_25d.py` | 微調整 |
| AC-07 | Dice 指標 | `metrics_segmentation.py` | 5-metric MICCAI suite に拡張 |
| AC-08 | 実験ノート | `docs/experiment_journey.md` の方法論 | 作業者が記載 |
| AC-09 | 公開結果と比較 | journey doc の Phase 4 "reality check" | 作業者分析 |
| AC-10 | 独自データ漏洩なし | `.gitignore` で `Datasets/`, weights, logs を除外 | 作業者規律 |

## クイックスタート

### 1. WMH データなしで継承パイプライン検証

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. 公開バンドルの内容確認

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. コードを触る前に inheritance map を読む

```bash
less docs/inheritance/inheritance_map.md
```

### 4. MONAI baseline で Phase 1 開始

kickoff brief 通り、AC-05 はまず MONAI / PyTorch 標準 3D segmentation モデル — 継承された異種アンサンブルは **使わない**。MONAI baseline が信頼できる Dice を出してから、継承された 2.5D / cross-arch を Phase 2 として活用。

## 同梱 / 除外

同梱:
- ソースコード（継承 Tier S 完全コピー + Tier A は DEFERRED_WMH_REVIEW stubs 付き）
- MPS 用 nnU-Net trainer variant
- Cross-architecture アンサンブルスクリプト
- 監査マップ、引用情報、roadmap、inheritance map
- 実験ジャーニー方法論ドキュメント

除外:
- `Datasets/` — WMH データは別途取得
- 学習済み重み (`*.pt`, `*.pth`)
- `runs/`, `results/`, `logs/`
- ISLES config と per-case 評価アーティファクト（意図的に除外）

## 引用方法

[CITATION.cff](CITATION.cff) 参照。本 repo の heterogeneous-ensemble または MPS 3D workaround コードを使用する場合は、上流の ISLES 姉妹 repo も引用すること。

## ライセンス

Apache License 2.0（コード）。[LICENSE](LICENSE) と [NOTICE](NOTICE) 参照。


## 2026-06 MLOps refactor delta

このパッケージは、公開WMH2017データをローカルに置いた状態で、以下の順に検証する構造へ更新した。

1. dataset manifest生成
2. label value audit
3. train/validation split生成
4. overlay可視化
5. MONAI 3D U-Net smoke training
6. validation prediction保存
7. local metric評価
8. run/evidence/release package manifest保存

最短コマンド:

```bash
python -m pip install -r requirements-lock.txt
export WMH2017_ROOT=/path/to/MICCAI2017_WMH/files
bash scripts/run_wmh2017_minimal_pipeline.sh
```

禁止:
- `challenge_split=test` をtraining/validation/threshold tuning/model selectionに使わない。
- `mask > 0` をforegroundにしない。foregroundは `label == 1` のみ。
- 独自データ、クラウド、顧客提示、臨床判断には使わない。
- official evaluation cross-checkなしにSOTA/leaderboard互換を主張しない。

現判定:
- 構造チェック: release_stateとは別管理
- 実データrun/evidenceと人間レビュー後: PREVIEW候補
- READY_FOR_RELEASEではない。
