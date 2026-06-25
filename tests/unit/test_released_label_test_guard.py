"""Default-off released-label local-test override: shared guard + loader/eval routing.

Covers the allow/block matrix for the shared guard, the load_case_records inference path, and that
threshold-tuning paths remain hard-blocked (no override). No training/eval run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wmh2017.config.training_config import resolve_input_modalities
from wmh2017.data.case_records import load_case_records
from wmh2017.data.split_guard import (
    guard_challenge_split_test,
    is_released_label_local_test_case,
)
from wmh2017.evaluation.threshold_sweep import _case_rows_for_split


# --- shared helper matrix ---------------------------------------------------
def test_non_test_always_passes() -> None:
    guard_challenge_split_test("c", "training", assigned_split="train")
    guard_challenge_split_test("c", "training", assigned_split="heldout_eval", allow_released_label_local_test=True)


def test_default_off_test_raises() -> None:
    with pytest.raises(ValueError, match="challenge_split=test"):
        guard_challenge_split_test("c", "test", assigned_split="heldout_eval")


def test_override_allows_only_heldout_eval() -> None:
    # Allowed: flag on + heldout_eval + test (released-label local test, inference or evaluation).
    guard_challenge_split_test("c", "test", assigned_split="heldout_eval", allow_released_label_local_test=True)
    # Blocked: flag on but train/val are never eligible.
    with pytest.raises(ValueError, match="challenge_split=test"):
        guard_challenge_split_test("c", "test", assigned_split="train", allow_released_label_local_test=True)
    with pytest.raises(ValueError, match="challenge_split=test"):
        guard_challenge_split_test("c", "test", assigned_split="val", allow_released_label_local_test=True)


def test_is_released_label_local_test_case_truth_table() -> None:
    assert is_released_label_local_test_case("test", "heldout_eval", allow_released_label_local_test=True) is True
    assert is_released_label_local_test_case("test", "heldout_eval", allow_released_label_local_test=False) is False
    assert is_released_label_local_test_case("test", "train", allow_released_label_local_test=True) is False
    assert is_released_label_local_test_case("training", "heldout_eval", allow_released_label_local_test=True) is False


# --- load_case_records (inference path) -------------------------------------
def _write(tmp_path: Path, assigned: str) -> tuple[Path, Path]:
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "case_id": "c1",
                "challenge_split": "test",
                "flair_pre_path": "/d/c1_flair.nii.gz",
                "t1_pre_path": "/d/c1_t1.nii.gz",
                "wmh_path": "/d/c1_wmh.nii.gz",
            }
        ]
    ).to_csv(manifest_csv, index=False)
    pd.DataFrame([{"case_id": "c1", "assigned_split": assigned}]).to_csv(split_csv, index=False)
    return manifest_csv, split_csv


def _modalities():
    return resolve_input_modalities(
        {
            "input_modalities": [
                {"name": "flair", "manifest_key": "flair_pre_path", "required": True},
                {"name": "t1", "manifest_key": "t1_pre_path", "required": True},
            ]
        }
    )


def test_load_case_records_heldout_allowed_with_flag(tmp_path: Path) -> None:
    manifest_csv, split_csv = _write(tmp_path, "heldout_eval")
    recs = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split="heldout_eval",
        input_modalities=_modalities(),
        label_key="wmh_path",
        allow_released_label_local_test=True,
    )
    assert [r.case_id for r in recs] == ["c1"]


def test_load_case_records_heldout_blocked_without_flag(tmp_path: Path) -> None:
    manifest_csv, split_csv = _write(tmp_path, "heldout_eval")
    with pytest.raises(ValueError, match="challenge_split=test"):
        load_case_records(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            assigned_split="heldout_eval",
            input_modalities=_modalities(),
            label_key="wmh_path",
        )


def test_load_case_records_val_blocked_even_with_flag(tmp_path: Path) -> None:
    manifest_csv, split_csv = _write(tmp_path, "val")
    with pytest.raises(ValueError, match="challenge_split=test"):
        load_case_records(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            assigned_split="val",
            input_modalities=_modalities(),
            label_key="wmh_path",
            allow_released_label_local_test=True,
        )


# --- threshold sweep stays hard-blocked (no override path) ------------------
def test_threshold_sweep_still_blocks_test(tmp_path: Path) -> None:
    manifest_csv, split_csv = _write(tmp_path, "heldout_eval")
    with pytest.raises(ValueError, match="challenge_split=test"):
        _case_rows_for_split(manifest_csv, split_csv, "heldout_eval")
