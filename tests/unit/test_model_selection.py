from __future__ import annotations

from pathlib import Path

import pytest

from wmh2017.evaluation.model_selection import (
    Candidate,
    CandidateSelectionRule,
    candidate_from_cv_summary,
    rule_from_config,
    select_candidate,
    write_candidate_selection_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RULE_CONFIG = REPO_ROOT / "configs" / "selection" / "candidate_dice_constrained_recall.yaml"

# Audited 5-fold CV means.
A2 = Candidate(
    cv_id="a2cv", metrics={"mean_dice": 0.6138, "mean_lesion_recall": 0.2071, "mean_lesion_f1": 0.2966}, n_folds=5
)
RC2 = Candidate(
    cv_id="rc2", metrics={"mean_dice": 0.6115, "mean_lesion_recall": 0.2718, "mean_lesion_f1": 0.3538}, n_folds=5
)


def _f1_rule(dice_floor: float) -> CandidateSelectionRule:
    return CandidateSelectionRule(
        primary_metric="mean_lesion_f1",
        primary_mode="max",
        constraints={"mean_dice_min": dice_floor},
        tie_breakers=("mean_lesion_recall", "mean_dice"),
    )


def test_dice_constrained_recall_selects_rc2():
    result = select_candidate([A2, RC2], _f1_rule(0.6088))
    assert result["selected"]["cv_id"] == "rc2"
    assert result["selection_scope"] == "candidate_family"
    assert [r["cv_id"] for r in result["ranked"]] == ["rc2", "a2cv"]
    assert result["excluded"] == []


def test_tighter_dice_floor_excludes_rc2_and_selects_a2():
    result = select_candidate([A2, RC2], _f1_rule(0.6120))
    assert result["selected"]["cv_id"] == "a2cv"
    assert [e["cv_id"] for e in result["excluded"]] == ["rc2"]
    assert "mean_dice" in result["excluded"][0]["violations"][0]


def test_all_excluded_raises():
    with pytest.raises(ValueError, match="no candidate satisfies constraints"):
        select_candidate([A2, RC2], _f1_rule(0.99))


def test_missing_metric_raises():
    bad = Candidate(cv_id="x", metrics={"mean_dice": 0.5}, n_folds=1)
    with pytest.raises(KeyError):
        select_candidate([bad], CandidateSelectionRule(primary_metric="mean_lesion_f1"))


def test_tie_on_primary_uses_tie_breaker():
    a = Candidate(cv_id="a", metrics={"mean_lesion_f1": 0.30, "mean_lesion_recall": 0.20})
    b = Candidate(cv_id="b", metrics={"mean_lesion_f1": 0.30, "mean_lesion_recall": 0.25})
    rule = CandidateSelectionRule(primary_metric="mean_lesion_f1", tie_breakers=("mean_lesion_recall",))
    assert select_candidate([a, b], rule)["selected"]["cv_id"] == "b"


def test_min_mode_prefers_lower_primary():
    a = Candidate(cv_id="a", metrics={"mean_hd95": 9.0})
    b = Candidate(cv_id="b", metrics={"mean_hd95": 4.0})
    rule = CandidateSelectionRule(primary_metric="mean_hd95", primary_mode="min")
    assert select_candidate([a, b], rule)["selected"]["cv_id"] == "b"


def test_artifact_hash_is_stable_and_present(tmp_path: Path):
    result = select_candidate([A2, RC2], _f1_rule(0.6088))
    out = tmp_path / "candidate_selection_result.json"
    payload = write_candidate_selection_artifact(out, result)
    assert out.exists()
    assert payload["artifact_hash"]
    # Re-writing the same result yields the same hash (rule + ranking bound).
    again = write_candidate_selection_artifact(tmp_path / "again.json", result)
    assert again["artifact_hash"] == payload["artifact_hash"]


def test_rule_from_real_config_parses():
    import yaml

    rule = rule_from_config(yaml.safe_load(RULE_CONFIG.read_text(encoding="utf-8")))
    assert rule.primary_metric == "mean_lesion_f1"
    assert rule.constraints == {"mean_dice_min": 0.6088}
    assert rule.tie_breakers == ("mean_lesion_recall", "mean_dice")


def test_candidate_from_cv_summary_extracts_means():
    summary = {
        "n_folds": 5,
        "metrics": {
            "mean_dice": {"mean": 0.6115, "std": 0.04, "n": 5},
            "mean_lesion_recall": {"mean": 0.2718, "std": 0.08, "n": 5},
            "mean_lesion_f1": {"mean": 0.3538, "std": 0.05, "n": 5},
        },
        "cv_id": "rc2",
    }
    cand = candidate_from_cv_summary(summary)
    assert cand.cv_id == "rc2"
    assert cand.n_folds == 5
    assert cand.metrics == {"mean_dice": 0.6115, "mean_lesion_recall": 0.2718, "mean_lesion_f1": 0.3538}


def test_invalid_primary_mode_raises():
    with pytest.raises(ValueError, match="unsupported primary_mode"):
        CandidateSelectionRule(primary_metric="mean_dice", primary_mode="sideways")
