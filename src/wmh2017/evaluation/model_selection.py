"""Cross-run candidate selection for WMH2017 (pre-registered Pareto rule).

This picks the best *experiment* (e.g. A2-CV vs RC2 vs T1 variants) under a rule that is
fixed before the results are read: one primary metric + explicit constraints (e.g. a Dice
floor) + tie-breakers. The chosen rule is echoed and hashed into the selection artifact so
the decision criterion cannot be edited after the fact (metric gaming guard).

This is deliberately distinct from the within-run checkpoint ``SelectionPolicy``
(``wmh2017.training.selection``): that chooses the best *epoch* inside one training run;
this chooses the best *run* across a candidate family. Both are local validation only;
selection_scope here is ``candidate_family`` and makes no SOTA/official/clinical claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wmh2017.evaluation.cv_aggregate import CV_METRICS
from wmh2017.lineage.hashes import sha256_jsonable, write_json

_CLAIM_BOUNDARY = "local cross-validation candidate selection only; not SOTA/official/clinical/production"
_PROHIBITED_USE = ["test split selection", "SOTA claim", "clinical decision", "production deployment"]


@dataclass(frozen=True)
class CandidateSelectionRule:
    """Pre-registered rule for choosing one experiment among several.

    ``constraints`` keys use a ``"<metric>_min"`` / ``"<metric>_max"`` suffix, e.g.
    ``{"mean_dice_min": 0.6088}`` keeps only candidates with mean_dice >= 0.6088.
    ``tie_breakers`` are always applied in max-order (higher is better).
    """

    primary_metric: str
    primary_mode: str = "max"
    constraints: Mapping[str, float] = field(default_factory=dict)
    tie_breakers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.primary_metric:
            raise ValueError("primary_metric must be non-empty")
        if self.primary_mode not in {"max", "min"}:
            raise ValueError(f"unsupported primary_mode: {self.primary_mode}; valid={{max, min}}")


@dataclass(frozen=True)
class Candidate:
    """One experiment's aggregated (cross-fold mean) metrics."""

    cv_id: str
    metrics: Mapping[str, float]
    n_folds: int = 0
    source: str = ""


def candidate_from_cv_summary(summary: Mapping[str, Any], *, cv_id: str | None = None) -> Candidate:
    """Build a Candidate from a cv_aggregate-shaped summary dict.

    Reads ``metrics[<m>]["mean"]`` for each CV metric present.
    """
    raw_metrics = summary.get("metrics", {})
    means = {m: float(raw_metrics[m]["mean"]) for m in CV_METRICS if m in raw_metrics}
    resolved_id = cv_id or str(summary.get("cv_id", "")) or "unknown"
    return Candidate(cv_id=resolved_id, metrics=means, n_folds=int(summary.get("n_folds", 0)))


def load_candidate(path: str | Path) -> Candidate:
    """Load a Candidate from a ``cv_summary_*.json`` file."""
    p = Path(path)
    summary = json.loads(p.read_text(encoding="utf-8"))
    cv_id = str(summary.get("cv_id", "")) or p.stem.replace("cv_summary_", "")
    return candidate_from_cv_summary(summary, cv_id=cv_id)


def _split_constraint_key(key: str) -> tuple[str, str]:
    if key.endswith("_min"):
        return key[:-4], "min"
    if key.endswith("_max"):
        return key[:-4], "max"
    raise ValueError(f"constraint key must end with _min or _max: {key}")


def _constraint_violations(metrics: Mapping[str, float], constraints: Mapping[str, float]) -> list[str]:
    """Return readable violation strings; raise KeyError for a missing constrained metric."""
    violations: list[str] = []
    for key, bound in constraints.items():
        metric, op = _split_constraint_key(key)
        if metric not in metrics:
            raise KeyError(f"constrained metric '{metric}' absent from candidate metrics {sorted(metrics)}")
        value = float(metrics[metric])
        if op == "min" and value < float(bound):
            violations.append(f"{metric}={value:.6g} < min {float(bound):.6g}")
        elif op == "max" and value > float(bound):
            violations.append(f"{metric}={value:.6g} > max {float(bound):.6g}")
    return violations


def _require_metric(candidate: Candidate, metric: str) -> float:
    if metric not in candidate.metrics:
        raise KeyError(f"candidate '{candidate.cv_id}' missing metric '{metric}' (has {sorted(candidate.metrics)})")
    return float(candidate.metrics[metric])


def _sort_key(candidate: Candidate, rule: CandidateSelectionRule) -> tuple[float, ...]:
    primary = _require_metric(candidate, rule.primary_metric)
    # Normalize so the whole key is "higher is better" and sortable with reverse=True.
    primary_adjusted = primary if rule.primary_mode == "max" else -primary
    tie_values = tuple(_require_metric(candidate, tb) for tb in rule.tie_breakers)
    return (primary_adjusted, *tie_values)


def rule_to_payload(rule: CandidateSelectionRule) -> dict[str, Any]:
    """Serialize the rule for the selection artifact."""
    return {
        "primary_metric": rule.primary_metric,
        "primary_mode": rule.primary_mode,
        "constraints": dict(rule.constraints),
        "tie_breakers": list(rule.tie_breakers),
    }


def rule_from_config(mapping: Mapping[str, Any]) -> CandidateSelectionRule:
    """Parse a ``candidate_selection:`` block (or its parent mapping) into a rule."""
    block = mapping.get("candidate_selection", mapping)
    raw_constraints = block.get("constraints", {}) or {}
    raw_tb = block.get("tie_breakers", ()) or ()
    return CandidateSelectionRule(
        primary_metric=str(block["primary_metric"]),
        primary_mode=str(block.get("primary_mode", "max")),
        constraints={str(k): float(v) for k, v in raw_constraints.items()},
        tie_breakers=tuple(str(x) for x in raw_tb),
    )


def _selection_policy_string(rule: CandidateSelectionRule) -> str:
    parts = [f"{rule.primary_mode} {rule.primary_metric}"]
    if rule.constraints:
        parts.append("subject to " + ", ".join(f"{k}={v:.6g}" for k, v in rule.constraints.items()))
    if rule.tie_breakers:
        parts.append("tie-break " + " then ".join(rule.tie_breakers))
    return "; ".join(parts)


def select_candidate(candidates: Sequence[Candidate], rule: CandidateSelectionRule) -> dict[str, Any]:
    """Rank candidates by the pre-registered rule and return an artifact-ready result."""
    if not candidates:
        raise ValueError("no candidates provided")

    eligible: list[Candidate] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        violations = _constraint_violations(candidate.metrics, rule.constraints)
        if violations:
            excluded.append({"cv_id": candidate.cv_id, "metrics": dict(candidate.metrics), "violations": violations})
        else:
            eligible.append(candidate)

    if not eligible:
        raise ValueError(
            f"no candidate satisfies constraints {dict(rule.constraints)}; "
            f"excluded={[e['cv_id'] for e in excluded]}"
        )

    ranked = sorted(eligible, key=lambda c: _sort_key(c, rule), reverse=True)
    ranked_payload = [
        {"rank": i + 1, "cv_id": c.cv_id, "n_folds": c.n_folds, "metrics": dict(c.metrics)}
        for i, c in enumerate(ranked)
    ]
    best = ranked[0]
    return {
        "selection_scope": "candidate_family",
        "selected": {"cv_id": best.cv_id, "n_folds": best.n_folds, "metrics": dict(best.metrics)},
        "ranked": ranked_payload,
        "excluded": excluded,
        "rule": rule_to_payload(rule),
        "selection_policy": _selection_policy_string(rule),
        "claim_boundary": _CLAIM_BOUNDARY,
        "allowed_use": "local cross-validation candidate selection",
        "prohibited_use": _PROHIBITED_USE,
    }


def write_candidate_selection_artifact(out_path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Hash and write the selection result; the hash binds the rule + ranking."""
    payload = dict(result)
    payload.pop("artifact_hash", None)
    payload["artifact_hash"] = sha256_jsonable(payload)
    write_json(out_path, payload)
    return payload
