"""k-fold split generation: coverage, no leakage, site stratification."""

from __future__ import annotations

import pandas as pd
import pytest

from wmh2017.data.splits import (
    KFoldPolicy,
    assert_kfold_coverage,
    assert_no_test_contamination,
    make_kfold_splits,
)


def _toy_manifest(n_per_site: int = 10, sites=("A", "B", "C")) -> pd.DataFrame:
    rows = []
    cid = 0
    for site in sites:
        for _ in range(n_per_site):
            rows.append({"case_id": f"c{cid}", "challenge_split": "training", "site": site})
            cid += 1
    # Add test rows that must never enter train/val.
    for _ in range(5):
        rows.append({"case_id": f"t{cid}", "challenge_split": "test", "site": "A"})
        cid += 1
    return pd.DataFrame(rows)


def test_folds_partition_training_cases() -> None:
    manifest = _toy_manifest()
    folds = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    assert len(folds) == 5
    assert_kfold_coverage(folds)


def test_each_case_in_exactly_one_val_fold() -> None:
    manifest = _toy_manifest()
    folds = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    val_counts: dict[str, int] = {}
    for f in folds:
        for cid in f[f["assigned_split"] == "val"]["case_id"]:
            val_counts[cid] = val_counts.get(cid, 0) + 1
    assert set(val_counts.values()) == {1}
    # 30 training cases across 5 folds.
    assert len(val_counts) == 30


def test_no_test_contamination_in_any_fold() -> None:
    manifest = _toy_manifest()
    folds = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    for f in folds:
        assert_no_test_contamination(f)
        test_rows = f[f["challenge_split"] == "test"]
        assert (test_rows["assigned_split"] == "heldout_eval").all()


def test_site_stratification_is_balanced() -> None:
    manifest = _toy_manifest(n_per_site=10)
    folds = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    for f in folds:
        val = f[f["assigned_split"] == "val"]
        counts = val.groupby("site").size().to_dict()
        # 10 per site / 5 folds = 2 per site per val fold.
        assert counts == {"A": 2, "B": 2, "C": 2}


def test_deterministic_for_fixed_seed() -> None:
    manifest = _toy_manifest()
    a = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    b = make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=5, seed=42))
    for fa, fb in zip(a, b, strict=True):
        pd.testing.assert_series_equal(
            fa.sort_values("case_id")["assigned_split"].reset_index(drop=True),
            fb.sort_values("case_id")["assigned_split"].reset_index(drop=True),
        )


def test_rejects_k_below_2() -> None:
    manifest = _toy_manifest()
    with pytest.raises(ValueError):
        make_kfold_splits(manifest, KFoldPolicy(split_id="X", k=1, seed=42))
