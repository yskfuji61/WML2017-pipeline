import pandas as pd

from wmh2017.data.splits import SplitPolicy, assert_no_test_contamination, make_train_val_split


def test_test_rows_are_heldout_eval_even_when_mask_exists():
    manifest = pd.DataFrame(
        {
            "case_id": ["tr1", "tr2", "te1"],
            "challenge_split": ["training", "training", "test"],
            "has_wmh": [True, True, True],
        }
    )
    split_df = make_train_val_split(manifest, SplitPolicy(split_id="T", seed=42, train_ratio=0.5))
    assert_no_test_contamination(split_df)
    test_row = split_df[split_df["case_id"] == "te1"].iloc[0]
    assert test_row["assigned_split"] == "heldout_eval"


def test_unknown_source_split_is_blocked():
    manifest = pd.DataFrame(
        {
            "case_id": ["tr1", "tr2", "u1"],
            "challenge_split": ["training", "training", "unknown"],
            "has_wmh": [True, True, True],
        }
    )
    split_df = make_train_val_split(manifest, SplitPolicy(split_id="T", seed=42, train_ratio=0.5))
    unknown_row = split_df[split_df["case_id"] == "u1"].iloc[0]
    assert unknown_row["assigned_split"] == "blocked_unknown_source_split"


def test_training_without_primary_wmh_is_not_eligible():
    manifest = pd.DataFrame(
        {
            "case_id": ["tr1", "tr2", "te1"],
            "challenge_split": ["training", "training", "test"],
            "has_wmh": [False, True, True],
        }
    )
    split_df = make_train_val_split(manifest, SplitPolicy(split_id="T", seed=42, train_ratio=0.5))
    eligible = split_df[split_df["assigned_split"].isin(["train", "val"])]
    assert set(eligible["case_id"]) == {"tr2"}
