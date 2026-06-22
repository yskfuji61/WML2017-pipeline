# Phase gate judgment — RC2 (5-fold CV)

**Recorded:** 2026-06-22 UTC  
**Commit:** `d37e613`  
**CV summary:** `reports/cv/cv_summary_rc2_seed42.json`  
**CV run_id:** `wmh2017_rc2_seed42`  
**Reference fold0 run_id:** `wmh2017_rc2_cosine_fold0_seed42`

## Gate definitions

| Gate | Dice threshold | Recall threshold | Purpose |
|------|---------------:|-----------------:|---------|
| Phase A | 0.65 | 0.35 | Minimum local validation bar |
| Phase B | 0.72 | — | Stronger Dice bar (recall not redefined) |

## RC2 5-fold CV results (primary)

| Metric | Mean ± std (n=5) | Phase A threshold | Judgment |
|--------|-----------------:|------------------:|----------|
| mean_dice | 0.612 ± 0.047 | 0.65 | **NOT met** |
| mean_lesion_recall | 0.272 ± 0.084 | 0.35 | **NOT met** |
| mean_lesion_f1 | 0.354 ± 0.056 | — | recorded |

Phase B (Dice 0.72): **NOT met** (mean_dice 0.612).

## Reference — fold 0 only (not primary claim)

| Metric | fold 0 value |
|--------|-------------:|
| mean_dice | 0.671 |
| mean_lesion_recall | 0.252 |

Use fold 0 for illustration only; CV mean is the honest unit of measurement.

## Comparison vs A2-CV (`wmh2017_a2cv_cosine_seed42`)

| Metric | A2-CV | RC2-CV | Delta |
|--------|------:|-------:|------:|
| mean_dice | 0.614 ± 0.037 | 0.612 ± 0.047 | −0.002 |
| mean_lesion_recall | 0.207 ± 0.038 | 0.272 ± 0.084 | **+0.065** |

Recall improved under the RC2 recipe (Tversky beta=0.85, pos=2) but remains below the Phase A recall gate.

## Claim boundary

- Local cross-validation only; test split never used.
- Phase A/B gates are **not met** — do not claim gate completion.
- No SOTA, official-benchmark, clinical, customer, production, or proprietary-data claims.
