"""All-train final-checkpoint support: row-guard + checkpoint-policy helpers.

These cover the default-off all-train path (empty val + last-epoch checkpoint, no validation- or
test-based selection). Pure-function tests: no torch/monai import, no training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh2017.training.train_monai import (
    _split_has_assigned,
    require_train_val_rows,
    resolve_checkpoint_policy,
)

_HEADER = (
    "split_id,fold,case_id,challenge_split,source_split,assigned_split,"
    "site,scanner,scanner_code,group_id,seed,reason,created_at\n"
)


def _write_split(path: Path, assigned: list[str]) -> Path:
    lines = [_HEADER]
    for i, a in enumerate(assigned):
        lines.append(f"S,0,{i},training,training,{a},Site,Sc,sc,{i},42,r,t\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_default_off_normal_rows_ok() -> None:
    # Default behavior unchanged: both train and val present, flags off.
    require_train_val_rows(["a"], ["b"], allow_empty_val=False, save_last_checkpoint=False)


def test_empty_val_without_allow_raises() -> None:
    with pytest.raises(ValueError, match="train and val rows are required"):
        require_train_val_rows(["a"], [], allow_empty_val=False, save_last_checkpoint=False)


def test_allow_empty_val_without_save_last_raises() -> None:
    with pytest.raises(ValueError, match="save_last_checkpoint"):
        require_train_val_rows(["a"], [], allow_empty_val=True, save_last_checkpoint=False)


def test_allow_empty_val_requires_empty_val() -> None:
    with pytest.raises(ValueError, match="empty val"):
        require_train_val_rows(["a"], ["b"], allow_empty_val=True, save_last_checkpoint=True)


def test_empty_train_always_raises() -> None:
    with pytest.raises(ValueError, match="train rows are required"):
        require_train_val_rows([], [], allow_empty_val=True, save_last_checkpoint=True)
    with pytest.raises(ValueError, match="train rows are required"):
        require_train_val_rows([], ["b"], allow_empty_val=False, save_last_checkpoint=False)


def test_alltrain_ok() -> None:
    # All-train mode: empty val, both flags set, train present -> no raise.
    require_train_val_rows(["a", "b"], [], allow_empty_val=True, save_last_checkpoint=True)


def test_policy_last_epoch_for_alltrain() -> None:
    assert resolve_checkpoint_policy(run_val=False, save_last_checkpoint=True) == "last_epoch"


def test_policy_best_on_val_for_cv() -> None:
    assert resolve_checkpoint_policy(run_val=True, save_last_checkpoint=False) == "best_on_val"
    # save_last alongside a real val set still selects best-on-val (last is only an extra artifact).
    assert resolve_checkpoint_policy(run_val=True, save_last_checkpoint=True) == "best_on_val"
    # No val and no save_last is not a valid all-train config, but the policy label is best_on_val.
    assert resolve_checkpoint_policy(run_val=False, save_last_checkpoint=False) == "best_on_val"


def test_split_has_assigned_detects_val(tmp_path: Path) -> None:
    cv = _write_split(tmp_path / "cv.csv", ["train", "train", "val", "heldout_eval"])
    assert _split_has_assigned(cv, "val") is True
    assert _split_has_assigned(cv, "train") is True
    assert _split_has_assigned(cv, "heldout_eval") is True


def test_split_has_assigned_false_for_alltrain(tmp_path: Path) -> None:
    alltrain = _write_split(tmp_path / "alltrain.csv", ["train", "train", "heldout_eval"])
    assert _split_has_assigned(alltrain, "val") is False
    assert _split_has_assigned(alltrain, "train") is True


def test_alltrain_prep_path_skips_val_and_passes_guard(tmp_path: Path) -> None:
    # Ties the loader peek to the guard: a val-less split + allow_empty_val + save_last_checkpoint
    # takes the empty-val branch (no load) and require_train_val_rows accepts the empty val set.
    alltrain = _write_split(tmp_path / "alltrain.csv", ["train", "train", "heldout_eval"])
    assert _split_has_assigned(alltrain, "val") is False
    require_train_val_rows(["a", "b"], [], allow_empty_val=True, save_last_checkpoint=True)


def test_normal_cv_split_would_load_val(tmp_path: Path) -> None:
    cv = _write_split(tmp_path / "cv.csv", ["train", "val"])
    # Normal CV: val present -> the loader peek is True, so the val load runs as before.
    assert _split_has_assigned(cv, "val") is True
