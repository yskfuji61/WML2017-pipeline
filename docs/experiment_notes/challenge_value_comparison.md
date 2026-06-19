# Challenge value comparison (local PoC reference only)

Last updated: 2026-06-19

## Claim boundary

This note compares **local validation smoke metrics** to **published report ranges** for orientation only.
It does **not** claim official hidden-test benchmark equivalence, leaderboard ranking, or clinical performance.

## Local reference (seed42 smoke wiring check)

Source: [`wmh2017_local_e2e_seed42.md`](wmh2017_local_e2e_seed42.md) / [`smoke_run_evidence_summary.md`](../../reports/learning_evidence/smoke_run_evidence_summary.md)

| metric | local smoke (n=2 val cases) |
|---|---:|
| mean_dice | 0.00105 |
| mean_hd95 | 127.58 |
| mean_lesion_recall | 0.931 |
| mean_lesion_f1 | 0.024 |

Training: 1 epoch, 2 steps, patch 32³, MONAI 3D U-Net (tiny channels).

## Published challenge / literature reference ranges (orientation)

Reported WMH segmentation Dice on challenge-style setups often falls roughly in the **0.6–0.8** range depending on
model, preprocessing, and evaluation protocol. Exact numbers vary by site, metric script, and train/val/test policy.

| reference type | typical Dice range | notes |
|---|---|---|
| Challenge / peer-reviewed reports | ~0.6–0.8 (varies) | Not directly comparable to 2-step smoke |
| Local smoke (this repo) | ~0.001 | Wiring validation only |

## Gap interpretation (why local smoke Dice is far below literature)

Candidate causes (non-exhaustive):

1. **Training budget:** 2 optimizer steps vs full multi-epoch training.
2. **Patch size:** 32³ random crops vs whole-volume or larger patches.
3. **Model capacity:** smoke U-Net (8/16/32 channels) vs full architectures.
4. **Preprocessing differences:** nonzero z-score only; no resampling harmonization vs other pipelines.
5. **Label policy:** foreground = label==1 only; label==2 ignored (WMH2017 policy).
6. **Evaluation scope:** 2 val cases with saved predictions vs full validation set.
7. **Metric implementation:** local `dice_wmh_label1` vs official evaluator (GAP-004/013 still open).

## Next checks before any performance discussion

- Run `training.mode: full` config locally with documented epochs and full-val inference.
- Record redacted metrics in experiment notes; keep raw `.nii.gz` / checkpoints gitignored.
- Obtain human approval before official evaluator fetch or external performance claims.
