# RC1–RC3 fold0 probe comparison (seed 42)

**Recorded:** 2026-06-22 UTC  
**Split:** k-fold fold 0 only (`data/splits/wmh2017_kfold_seed42/fold0.csv`)  
**Claim boundary:** Local validation only; fold0 probe for variant selection — not a CV performance claim. Primary measurement is 5-fold CV in `cv_summary_rc2_seed42.json`.

## Config differences

| Variant | Config | Tversky (alpha/beta) | pos sampling | run_id |
|---------|--------|---------------------:|--------------|--------|
| RC1 | [`exp_rc1_cosine_fold0.yaml`](../../configs/experiments/recall/exp_rc1_cosine_fold0.yaml) | 0.20 / **0.80** | pos=2 | `wmh2017_rc1_cosine_fold0_seed42` |
| RC2 | [`exp_rc2_cosine_fold0.yaml`](../../configs/experiments/recall/exp_rc2_cosine_fold0.yaml) | 0.15 / **0.85** | pos=2 | `wmh2017_rc2_cosine_fold0_seed42` |
| RC3 | [`exp_rc3_cosine_fold0.yaml`](../../configs/experiments/recall/exp_rc3_cosine_fold0.yaml) | 0.20 / 0.80 | pos=1 (default) | `wmh2017_rc3_cosine_fold0_seed42` |

All variants: TverskyFocal gamma=1.33, cosine LR, 100 epochs, selection_metric=mean_dice (tie-break: recall, f1).

## fold0 validation metrics (post threshold sweep)

| Variant | mean_dice | mean_lesion_recall | mean_lesion_f1 | Selection |
|---------|----------:|-------------------:|---------------:|-----------|
| RC1 | 0.665 | 0.228 | 0.322 | — |
| **RC2** | 0.664 | **0.243** | **0.331** | **selected** |
| RC3 | 0.650 | 0.168 | 0.242 | rejected |

Sources:

- [`cv_summary_rc1_probe.json`](cv_summary_rc1_probe.json)
- [`cv_summary_rc2_probe.json`](cv_summary_rc2_probe.json)
- [`cv_summary_rc3_probe.json`](cv_summary_rc3_probe.json)

## Selection rationale

1. **RC2 selected** — highest recall (+0.015 vs RC1) with negligible Dice regression (−0.001 vs RC1).
2. **RC3 rejected** — loss-only FN weight without positive sampling hurt recall (0.168 vs RC1 0.228).
3. Full **5-fold CV** with RC2 recipe followed; see [`cv_summary_rc2_seed42.json`](cv_summary_rc2_seed42.json) and [`phase_gate_judgment_rc2.md`](../../docs/release_evidence/phase_gate_judgment_rc2.md).

Phase A gate (Dice 0.65 / Recall 0.35) and Phase B gate (Dice 0.72) remain **NOT met** after 5-fold CV.
