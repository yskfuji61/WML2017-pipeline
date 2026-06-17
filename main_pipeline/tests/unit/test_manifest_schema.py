import pandas as pd

REQUIRED_COLUMNS = {
    "dataset_id",
    "case_id",
    "challenge_split",
    "source_split",
    "site",
    "scanner",
    "scanner_code",
    "case_dir",
    "flair_pre_path",
    "t1_pre_path",
    "wmh_path",
    "has_wmh",
    "has_additional_o3",
    "has_additional_o4",
    "flair_path",
    "t1_path",
    "mask_path",
}


def test_dataset_manifest_required_columns():
    df = pd.DataFrame([{
        "dataset_id": "WMH2017",
        "case_id": "case001",
        "challenge_split": "training",
        "source_split": "training",
        "site": "Utrecht",
        "scanner": "3T Philips Achieva",
        "scanner_code": "utrecht_philips_achieva_3t",
        "case_dir": "/tmp/training/Utrecht/0",
        "flair_pre_path": "/tmp/training/Utrecht/0/pre/FLAIR.nii.gz",
        "t1_pre_path": "/tmp/training/Utrecht/0/pre/T1.nii.gz",
        "wmh_path": "/tmp/training/Utrecht/0/wmh.nii.gz",
        "has_wmh": True,
        "has_additional_o3": False,
        "has_additional_o4": False,
        "flair_path": "/tmp/training/Utrecht/0/pre/FLAIR.nii.gz",
        "t1_path": "/tmp/training/Utrecht/0/pre/T1.nii.gz",
        "mask_path": "/tmp/training/Utrecht/0/wmh.nii.gz",
    }])
    assert REQUIRED_COLUMNS.issubset(df.columns)
