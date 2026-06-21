"""Cross-validation aggregation for WMH2017 local validation.

Aggregates per-fold validation metrics into mean +/- std. This is local
cross-validation only: it must never consume test-split metrics, and it makes no
SOTA/official/clinical/production claim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

CV_METRICS = ("mean_dice", "mean_lesion_recall", "mean_lesion_f1")


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(values) / n
    # Sample standard deviation (ddof=1) when n > 1, else 0.0.
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = sqrt(var)
    else:
        std = 0.0
    return {"mean": float(mean), "std": float(std), "n": int(n)}


def _assert_validation_only(summary: dict[str, Any], *, source: str) -> None:
    assigned = str(summary.get("assigned_split", "val")).lower()
    if assigned not in {"val", "validation"}:
        raise ValueError(f"cv_aggregate refuses non-validation metrics from {source}: assigned_split={assigned}")
    claim = summary.get("claim_allowed", {})
    if isinstance(claim, dict) and claim.get("leaderboard_or_sota", False):
        raise ValueError(f"cv_aggregate refuses leaderboard/SOTA-claimed metrics from {source}")


def aggregate_fold_summaries(
    fold_summaries: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] = CV_METRICS,
) -> dict[str, Any]:
    """Aggregate a list of per-fold metrics_summary dicts into mean +/- std."""
    if not fold_summaries:
        raise ValueError("no fold summaries provided")

    per_fold: list[dict[str, Any]] = []
    aggregated: dict[str, dict[str, float]] = {}
    for i, summary in enumerate(fold_summaries):
        _assert_validation_only(summary, source=f"fold[{i}]")
        per_fold.append(
            {
                "fold": int(summary.get("fold", i)),
                "n_cases": int(summary.get("n_cases", 0)),
                **{m: float(summary[m]) for m in metrics if m in summary},
            }
        )

    for m in metrics:
        values = [float(s[m]) for s in fold_summaries if m in s]
        aggregated[m] = _mean_std(values)

    return {
        "n_folds": int(len(fold_summaries)),
        "metrics": aggregated,
        "per_fold": per_fold,
        "claim_boundary": "local cross-validation only; not SOTA/official/clinical/production",
        "selection_note": "aggregated from validation-only per-fold metrics; test split never used",
    }


def collect_fold_summaries(run_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read each run's evaluation/metrics_summary.json into a list of dicts."""
    summaries: list[dict[str, Any]] = []
    for i, run_dir in enumerate(run_dirs):
        path = Path(run_dir) / "evaluation" / "metrics_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"metrics_summary.json missing for fold run: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary.setdefault("fold", i)
        summaries.append(summary)
    return summaries


def write_cv_summary(
    out_path: str | Path,
    fold_summaries: Sequence[dict[str, Any]],
    *,
    cv_id: str = "",
    metrics: Sequence[str] = CV_METRICS,
) -> dict[str, Any]:
    """Aggregate and write cv_summary.json."""
    payload = aggregate_fold_summaries(fold_summaries, metrics=metrics)
    if cv_id:
        payload["cv_id"] = cv_id
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
