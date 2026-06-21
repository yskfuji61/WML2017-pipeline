"""CV aggregation: mean +/- std and validation-only guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh2017.evaluation.cv_aggregate import (
    aggregate_fold_summaries,
    collect_fold_summaries,
    write_cv_summary,
)


def _summary(fold: int, dice: float, recall: float, f1: float) -> dict:
    return {
        "fold": fold,
        "assigned_split": "val",
        "n_cases": 12,
        "mean_dice": dice,
        "mean_lesion_recall": recall,
        "mean_lesion_f1": f1,
    }


def test_aggregate_mean_and_std() -> None:
    summaries = [
        _summary(0, 0.60, 0.30, 0.40),
        _summary(1, 0.70, 0.40, 0.50),
        _summary(2, 0.80, 0.50, 0.60),
    ]
    out = aggregate_fold_summaries(summaries)
    assert out["n_folds"] == 3
    assert out["metrics"]["mean_dice"]["mean"] == pytest.approx(0.70)
    # sample std (ddof=1) of [0.6,0.7,0.8] = 0.1
    assert out["metrics"]["mean_dice"]["std"] == pytest.approx(0.1)
    assert out["metrics"]["mean_lesion_recall"]["mean"] == pytest.approx(0.40)


def test_single_fold_std_zero() -> None:
    out = aggregate_fold_summaries([_summary(0, 0.5, 0.2, 0.3)])
    assert out["metrics"]["mean_dice"]["std"] == 0.0
    assert out["metrics"]["mean_dice"]["n"] == 1


def test_rejects_non_validation_metrics() -> None:
    bad = _summary(0, 0.5, 0.2, 0.3)
    bad["assigned_split"] = "test"
    with pytest.raises(ValueError):
        aggregate_fold_summaries([bad])


def test_rejects_leaderboard_claimed_metrics() -> None:
    bad = _summary(0, 0.5, 0.2, 0.3)
    bad["claim_allowed"] = {"leaderboard_or_sota": True}
    with pytest.raises(ValueError):
        aggregate_fold_summaries([bad])


def test_collect_and_write_roundtrip(tmp_path: Path) -> None:
    run_dirs = []
    for i in range(3):
        d = tmp_path / f"fold{i}"
        (d / "evaluation").mkdir(parents=True)
        (d / "evaluation" / "metrics_summary.json").write_text(
            json.dumps(_summary(i, 0.6 + 0.1 * i, 0.3, 0.4)), encoding="utf-8"
        )
        run_dirs.append(d)
    summaries = collect_fold_summaries(run_dirs)
    assert len(summaries) == 3
    out_path = tmp_path / "cv_summary.json"
    payload = write_cv_summary(out_path, summaries, cv_id="test_cv")
    assert out_path.exists()
    assert payload["cv_id"] == "test_cv"
    assert payload["metrics"]["mean_dice"]["n"] == 3


def test_collect_missing_summary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_fold_summaries([tmp_path / "nonexistent"])
