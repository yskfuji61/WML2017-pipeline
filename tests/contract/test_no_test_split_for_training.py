import pandas as pd

from wmh2017.data.splits import SplitPolicy, make_train_val_split


def test_challenge_test_split_is_never_used_for_training():
    manifest = pd.DataFrame(
        {
            "case_id": ["tr1", "tr2", "te1"],
            "challenge_split": ["training", "training", "test"],
            "has_wmh": [True, True, True],
        }
    )
    split_df = make_train_val_split(manifest, SplitPolicy(split_id="T", seed=42, train_ratio=0.5))
    train_cases = set(split_df[split_df["assigned_split"].isin(["train", "val"])]["case_id"])
    forbidden = set(split_df[split_df["challenge_split"] == "test"]["case_id"])
    assert train_cases.isdisjoint(forbidden)
