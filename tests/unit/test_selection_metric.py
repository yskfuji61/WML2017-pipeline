"""Unit tests for checkpoint selection policy primitives."""

from __future__ import annotations

import pytest

from wmh2017.training.selection import (
    DEFAULT_COMPOSITE_WEIGHTS,
    VALID_SELECTION_METRICS,
    SelectionPolicy,
    composite_dice_recall,
    evaluate_candidate,
    is_improved,
    policy_to_payload,
    resolve_selection_score,
    selection_policy_from_config,
)


def test_valid_metrics_set() -> None:
    assert VALID_SELECTION_METRICS == {
        "mean_dice",
        "mean_lesion_recall",
        "mean_lesion_f1",
        "val_loss_proxy",
        "composite_dice_recall",
    }


def test_composite_dice_recall_weighting() -> None:
    metrics = {"mean_dice": 0.8, "mean_lesion_recall": 0.4}
    assert composite_dice_recall(metrics, 0.7, 0.3) == pytest.approx(0.8 * 0.7 + 0.4 * 0.3)
    # equal weights average
    assert composite_dice_recall(metrics, 1.0, 1.0) == pytest.approx(0.6)


def test_composite_rejects_invalid_weights() -> None:
    metrics = {"mean_dice": 0.5, "mean_lesion_recall": 0.5}
    with pytest.raises(ValueError):
        composite_dice_recall(metrics, -0.1, 0.5)
    with pytest.raises(ValueError):
        composite_dice_recall(metrics, 0.0, 0.0)


def test_resolve_score_uses_default_composite_weights() -> None:
    metrics = {"mean_dice": 1.0, "mean_lesion_recall": 0.0}
    score = resolve_selection_score(metrics, metric="composite_dice_recall")
    assert score == pytest.approx(DEFAULT_COMPOSITE_WEIGHTS["mean_dice"])


def test_resolve_score_missing_metric_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        resolve_selection_score({"mean_dice": 0.5}, metric="mean_lesion_recall")


def test_resolve_score_unknown_metric_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        resolve_selection_score({"x": 1.0}, metric="not_a_metric")


def test_is_improved_max_and_min() -> None:
    assert is_improved(0.6, None) is True
    assert is_improved(0.6, 0.5, mode="max") is True
    assert is_improved(0.5, 0.5, mode="max") is False
    assert is_improved(0.4, 0.5, mode="min") is True
    assert is_improved(0.6, 0.5, mode="min") is False


def test_is_improved_min_delta() -> None:
    assert is_improved(0.51, 0.50, mode="max", min_delta=0.02) is False
    assert is_improved(0.53, 0.50, mode="max", min_delta=0.02) is True


def test_is_improved_rejects_bad_mode() -> None:
    with pytest.raises(ValueError):
        is_improved(0.5, 0.4, mode="sideways")


def test_policy_rejects_invalid_metric() -> None:
    with pytest.raises(ValueError):
        SelectionPolicy(metric="bogus")


def test_evaluate_candidate_tie_break_on_recall() -> None:
    policy = SelectionPolicy(metric="mean_dice", tie_breakers=("mean_lesion_recall",))
    best_metrics = {"mean_dice": 0.7, "mean_lesion_recall": 0.3, "mean_lesion_f1": 0.2}
    candidate = {"mean_dice": 0.7, "mean_lesion_recall": 0.5, "mean_lesion_f1": 0.2}
    decision = evaluate_candidate(policy, candidate, best_score=0.7, best_metrics=best_metrics)
    assert decision.improved is True
    assert "tie" in decision.reason


def test_policy_to_payload_roundtrip() -> None:
    policy = selection_policy_from_config(
        {
            "selection_metric": "mean_lesion_recall",
            "selection_tie_breakers": ["mean_dice"],
        }
    )
    payload = policy_to_payload(policy, checkpoint_semantics="x")
    assert payload["selection_metric"] == "mean_lesion_recall"
    assert payload["selection_mode"] == "max"
    assert payload["tie_breakers"] == ["mean_dice"]
    assert payload["checkpoint_semantics"] == "x"
