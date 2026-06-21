"""Checkpoint selection policy primitives for WMH2017 training.

This module centralizes "what does best mean" so that Dice-best, lesion-recall-best,
validation-loss-proxy-best, and composite-best cannot be confused across training
paths. It performs no I/O and makes no performance claim. Selecting a different
``selection_metric`` only changes which snapshot of a single training trajectory is
retained; it does not change the model's achievable performance frontier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

VALID_SELECTION_METRICS = {
    "mean_dice",
    "mean_lesion_recall",
    "mean_lesion_f1",
    "val_loss_proxy",
    "composite_dice_recall",
}

# Metrics whose "better" direction is minimization.
_MIN_METRICS = {"val_loss_proxy"}

DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {"mean_dice": 0.7, "mean_lesion_recall": 0.3}


@dataclass(frozen=True)
class SelectionPolicy:
    """Declarative checkpoint selection policy."""

    metric: str = "mean_dice"
    mode: str = "max"
    min_delta: float = 0.0
    tie_breakers: tuple[str, ...] = ()
    checkpoint_prefix: str = "model_best"
    composite_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.metric not in VALID_SELECTION_METRICS:
            raise ValueError(f"unsupported selection metric: {self.metric}; valid={sorted(VALID_SELECTION_METRICS)}")
        if self.mode not in {"max", "min"}:
            raise ValueError(f"unsupported selection mode: {self.mode}")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")


@dataclass(frozen=True)
class SelectionDecision:
    """Outcome of comparing a candidate score to the current best."""

    improved: bool
    score: float
    previous_best: float | None
    reason: str
    metrics: Mapping[str, float] = field(default_factory=dict)


def default_mode_for_metric(metric: str) -> str:
    """Return the natural optimization direction for a metric."""
    return "min" if metric in _MIN_METRICS else "max"


def composite_dice_recall(
    metrics: Mapping[str, float],
    dice_weight: float,
    recall_weight: float,
) -> float:
    """Weighted average of mean_dice and mean_lesion_recall."""
    if dice_weight < 0 or recall_weight < 0:
        raise ValueError("composite weights must be non-negative")
    total = dice_weight + recall_weight
    if total <= 0:
        raise ValueError("at least one composite weight must be positive")
    return (dice_weight * float(metrics["mean_dice"]) + recall_weight * float(metrics["mean_lesion_recall"])) / total


def resolve_selection_score(
    metrics: Mapping[str, float],
    *,
    metric: str,
    composite_weights: Mapping[str, float] | None = None,
) -> float:
    """Resolve the scalar selection score from a metrics mapping.

    Never silently falls back: an unknown or missing metric raises.
    """
    if metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"unsupported selection metric: {metric}")
    if metric == "composite_dice_recall":
        weights = composite_weights or DEFAULT_COMPOSITE_WEIGHTS
        return composite_dice_recall(
            metrics,
            dice_weight=float(weights["mean_dice"]),
            recall_weight=float(weights["mean_lesion_recall"]),
        )
    if metric not in metrics:
        raise KeyError(f"selection metric not found in metrics: {metric}")
    return float(metrics[metric])


def is_improved(
    score: float,
    best_score: float | None,
    *,
    mode: str = "max",
    min_delta: float = 0.0,
) -> bool:
    """Return True if ``score`` improves on ``best_score`` for the given mode."""
    if best_score is None:
        return True
    if mode == "max":
        return score > best_score + min_delta
    if mode == "min":
        return score < best_score - min_delta
    raise ValueError(f"unsupported selection mode: {mode}")


def _tie_break_value(metrics: Mapping[str, float], metric: str) -> float:
    """Tie-breaker comparison value; minimized metrics are negated for max-ordering."""
    value = float(metrics.get(metric, float("-inf")))
    return -value if metric in _MIN_METRICS else value


def evaluate_candidate(
    policy: SelectionPolicy,
    metrics: Mapping[str, float],
    *,
    best_score: float | None,
    best_metrics: Mapping[str, float] | None = None,
) -> SelectionDecision:
    """Decide whether ``metrics`` should become the new best under ``policy``.

    Primary comparison uses ``policy.metric``/``policy.mode``. If the primary score
    is tied with the current best (within strict equality), ``policy.tie_breakers``
    are applied in order using max-ordering (loss-like metrics negated).
    """
    score = resolve_selection_score(metrics, metric=policy.metric, composite_weights=policy.composite_weights)
    if is_improved(score, best_score, mode=policy.mode, min_delta=policy.min_delta):
        return SelectionDecision(
            improved=True,
            score=score,
            previous_best=best_score,
            reason=f"{policy.metric} improved ({policy.mode})",
            metrics=dict(metrics),
        )

    if best_score is not None and best_metrics is not None and score == best_score and policy.tie_breakers:
        for tb in policy.tie_breakers:
            cand = _tie_break_value(metrics, tb)
            prev = _tie_break_value(best_metrics, tb)
            if cand > prev:
                return SelectionDecision(
                    improved=True,
                    score=score,
                    previous_best=best_score,
                    reason=f"tie on {policy.metric}; {tb} improved",
                    metrics=dict(metrics),
                )
            if cand < prev:
                break

    return SelectionDecision(
        improved=False,
        score=score,
        previous_best=best_score,
        reason=f"{policy.metric} not improved",
        metrics=dict(metrics),
    )


def selection_policy_from_config(
    train_cfg: Mapping[str, Any],
    *,
    default_metric: str = "mean_dice",
    default_prefix: str = "model_best",
) -> SelectionPolicy:
    """Build a SelectionPolicy from a training-config mapping (backward compatible)."""
    metric = str(train_cfg.get("selection_metric", default_metric))
    mode = str(train_cfg.get("selection_mode", default_mode_for_metric(metric)))
    min_delta = float(train_cfg.get("selection_min_delta", 0.0))
    raw_tb = train_cfg.get("selection_tie_breakers", ()) or ()
    tie_breakers = tuple(str(x) for x in raw_tb) if isinstance(raw_tb, list | tuple) else ()
    prefix = str(train_cfg.get("checkpoint_prefix", default_prefix))
    raw_weights = train_cfg.get("composite_weights")
    composite_weights = {str(k): float(v) for k, v in raw_weights.items()} if isinstance(raw_weights, Mapping) else None
    return SelectionPolicy(
        metric=metric,
        mode=mode,
        min_delta=min_delta,
        tie_breakers=tie_breakers,
        checkpoint_prefix=prefix,
        composite_weights=composite_weights,
    )


def policy_to_payload(policy: SelectionPolicy, *, checkpoint_semantics: str) -> dict[str, object]:
    """Serialize a SelectionPolicy for evidence/checkpoint payloads."""
    return {
        "selection_metric": policy.metric,
        "selection_mode": policy.mode,
        "selection_min_delta": policy.min_delta,
        "tie_breakers": list(policy.tie_breakers),
        "composite_weights": dict(policy.composite_weights) if policy.composite_weights else None,
        "checkpoint_semantics": checkpoint_semantics,
    }
