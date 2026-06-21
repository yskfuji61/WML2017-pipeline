# ADR-0007: Metric selection and checkpoint semantics

## Status

Accepted

## Context

Checkpoint selection semantics were implicit and inconsistent across training paths,
making "best" ambiguous and hard to audit:

- MONAI training selected the best checkpoint by `val_dice` only.
- ConvNeXt 2.5D saved `model_best.pt` selected by a validation loss proxy
  (minimization), not by Dice or lesion recall.
- Threshold sweep selected a threshold by `mean_dice` with `mean_lesion_recall` then
  `mean_lesion_f1` tie-breakers, which is unrelated to checkpoint selection.
- E2E evaluation always passed `--skip-missing-predictions`, so full-run case
  coverage could be silently incomplete.

This created confusable situations: "recall improved but best did not update",
"`model_best.pt` is not Dice-best", and "threshold-sweep-best differs from
checkpoint-best".

## Decision

- Add a shared `src/wmh2017/training/selection.py` with explicit
  `selection_metric` / `selection_mode` and valid metrics: `mean_dice`,
  `mean_lesion_recall`, `mean_lesion_f1`, `val_loss_proxy`, `composite_dice_recall`.
  Missing metrics raise instead of silently falling back.
- MONAI training is driven by `selection_metric` (default `mean_dice`, `max`),
  computes lesion recall/F1 at validation, and writes `selection_policy`,
  `best_selection_score`, `best_selection_epoch`, `best_metrics`, and
  `checkpoint_semantics` into the checkpoint payload and `run_evidence.json`.
  A metric-explicit alias (`model_best_<metric>.pt`) is saved alongside the legacy
  `model_best.pt`.
- ConvNeXt 2.5D writes `model_best_val_loss_proxy.pt` as the primary checkpoint and
  keeps `model_best.pt` as a legacy alias; evidence records
  `selection_metric: val_loss_proxy` (`min`) and `metric_limitations`.
- Threshold sweep keeps its default policy but records
  `threshold_best_is_checkpoint_best: false`, `checkpoint_selection_metric`,
  `threshold_selection_metric`, `threshold_tie_breakers`, `allowed_use`, and
  `prohibited_use`. Threshold sweep never mutates checkpoint selection.
- E2E adds `--skip-missing-predictions` only for non-full (smoke) runs. Full
  evaluation fails if any expected prediction is missing, and `prediction_coverage`
  (expected/evaluated/missing) is recorded in the metrics summary.

## Non-goals

- No SOTA, official-benchmark, leaderboard-equivalence, clinical, customer, or
  production claim.
- No test-split tuning: test split is never used for selection, threshold tuning, or
  early stopping.
- No performance improvement is claimed from this change; it is governance and
  auditability only.

## Consequences

- Existing artifacts remain valid as legacy; backward-compatible fields
  (`best_val_dice`, `model_best.pt`) are retained but marked legacy.
- New artifacts must carry `selection_policy`, so `selection_metric` alone explains
  why a checkpoint is "best".
- Full E2E runs fail loudly on incomplete prediction coverage.

## Alternatives

- Keep implicit per-path semantics (rejected: not auditable, confusable).
- Hard-rename `model_best.pt` without an alias (rejected: breaks existing tooling).

## Reversal plan

Selection-policy changes require an ADR update. Revert is limited to
source/config/test/docs; data, artifacts, and evidence history are not rewritten.
