"""MONAI checkpoint selection semantics: metric drives best update."""

from __future__ import annotations

from wmh2017.training.selection import (
    evaluate_candidate,
    selection_policy_from_config,
)


def test_default_policy_is_mean_dice_max() -> None:
    policy = selection_policy_from_config({}, default_metric="mean_dice")
    assert policy.metric == "mean_dice"
    assert policy.mode == "max"


def test_recall_only_improvement_does_not_update_mean_dice_best() -> None:
    policy = selection_policy_from_config({"selection_metric": "mean_dice"})
    best_metrics = {"mean_dice": 0.70, "mean_lesion_recall": 0.30, "mean_lesion_f1": 0.40}
    # Recall improves, but mean_dice is unchanged -> default policy must NOT update.
    candidate = {"mean_dice": 0.70, "mean_lesion_recall": 0.60, "mean_lesion_f1": 0.40}
    decision = evaluate_candidate(policy, candidate, best_score=0.70, best_metrics=best_metrics)
    assert decision.improved is False


def test_recall_policy_updates_on_recall_improvement() -> None:
    policy = selection_policy_from_config({"selection_metric": "mean_lesion_recall"})
    best_metrics = {"mean_dice": 0.70, "mean_lesion_recall": 0.30, "mean_lesion_f1": 0.40}
    candidate = {"mean_dice": 0.65, "mean_lesion_recall": 0.55, "mean_lesion_f1": 0.40}
    decision = evaluate_candidate(policy, candidate, best_score=0.30, best_metrics=best_metrics)
    assert decision.improved is True
    assert decision.score == 0.55


def test_first_candidate_always_improves() -> None:
    policy = selection_policy_from_config({"selection_metric": "mean_dice"})
    candidate = {"mean_dice": 0.10, "mean_lesion_recall": 0.0, "mean_lesion_f1": 0.0}
    decision = evaluate_candidate(policy, candidate, best_score=None, best_metrics=None)
    assert decision.improved is True
