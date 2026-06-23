from __future__ import annotations

from pathlib import Path

import pandas as pd

from wmh2017.config.training_config import InputModality
from wmh2017.data.case_records import case_records_to_monai_rows, load_case_records


def _write_csvs(tmp_path: Path, manifest_rows: list[dict], split_rows: list[dict]) -> tuple[Path, Path]:
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    return manifest_csv, split_csv


FLAIR_T1 = (
    InputModality(name="flair", manifest_key="flair_pre_path", required=True),
    InputModality(name="t1", manifest_key="t1_pre_path", required=True),
)


def test_two_modalities_resolve_both_paths(tmp_path: Path):
    manifest_csv, split_csv = _write_csvs(
        tmp_path,
        [
            {
                "case_id": "c1",
                "challenge_split": "training",
                "flair_pre_path": "/data/c1_flair.nii.gz",
                "t1_pre_path": "/data/c1_t1.nii.gz",
                "wmh_path": "/data/c1_wmh.nii.gz",
            }
        ],
        [{"case_id": "c1", "assigned_split": "train"}],
    )
    records = load_case_records(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        assigned_split="train",
        input_modalities=FLAIR_T1,
        label_key="wmh_path",
    )
    assert records[0].image_paths == {
        "flair": "/data/c1_flair.nii.gz",
        "t1": "/data/c1_t1.nii.gz",
    }
    rows = case_records_to_monai_rows(records)
    assert rows == [
        {
            "case_id": "c1",
            "label": "/data/c1_wmh.nii.gz",
            "flair": "/data/c1_flair.nii.gz",
            "t1": "/data/c1_t1.nii.gz",
        }
    ]
