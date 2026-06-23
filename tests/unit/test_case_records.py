from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wmh2017.config.training_config import InputModality, resolve_input_modalities
from wmh2017.data.case_records import (
    case_records_to_monai_rows,
    load_case_records,
)


def _write_csvs(tmp_path: Path, manifest_rows: list[dict], split_rows: list[dict]) -> tuple[Path, Path]:
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    return manifest_csv, split_csv


def test_single_modality_rows_match_legacy_shape(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "training",
                "flair_pre_path": "/data/c1_flair.nii.gz",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "train"}],
    )
    modalities = resolve_input_modalities({"image_key": "flair_pre_path"})
    records = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split="train",
        input_modalities=modalities,
        label_key="wmh_path",
    )
    rows = case_records_to_monai_rows(records)
    assert rows == [{"case_id": "c1", "image": "/data/c1_flair.nii.gz", "label": "/data/c1_wmh.nii.gz"}]


def test_legacy_flair_fallback_when_image_key_empty(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "training",
                "flair_pre_path": "",
                "flair_path": "/data/fallback.nii.gz",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "val"}],
    )
    modalities = resolve_input_modalities({"image_key": "flair_pre_path"})
    records = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split="val",
        input_modalities=modalities,
        label_key="wmh_path",
    )
    assert records[0].image_paths["image"] == "/data/fallback.nii.gz"


def test_test_split_case_is_rejected(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "test",
                "flair_pre_path": "/data/c1.nii.gz",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "val"}],
    )
    modalities = resolve_input_modalities({"image_key": "flair_pre_path"})
    with pytest.raises(ValueError, match="challenge_split=test"):
        load_case_records(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            assigned_split="val",
            input_modalities=modalities,
            label_key="wmh_path",
        )


def test_missing_required_modality_raises(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "training",
                "flair_pre_path": "/data/c1_flair.nii.gz",
                "t1_pre_path": "",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "train"}],
    )
    modalities = (
        InputModality(name="flair", manifest_key="flair_pre_path"),
        InputModality(name="t1", manifest_key="t1_pre_path", required=True),
    )
    with pytest.raises(ValueError, match="missing required modality 't1'"):
        load_case_records(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            assigned_split="train",
            input_modalities=modalities,
            label_key="wmh_path",
        )


def test_optional_modality_missing_is_allowed(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "training",
                "flair_pre_path": "/data/c1_flair.nii.gz",
                "t1_pre_path": "",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "train"}],
    )
    modalities = (
        InputModality(name="flair", manifest_key="flair_pre_path"),
        InputModality(name="t1", manifest_key="t1_pre_path", required=False),
    )
    records = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split="train",
        input_modalities=modalities,
        label_key="wmh_path",
    )
    assert records[0].image_paths == {"flair": "/data/c1_flair.nii.gz", "t1": ""}
