import pandas as pd

from wmh2017.schemas import manifest_json_from_csv, redact_path


def test_redact_path():
    assert redact_path("/Users/secret/data.nii.gz") == "REDACTED_OR_LOCAL_ONLY"
    assert redact_path("") == "REDACTED_OR_LOCAL_ONLY_OR_NULL"


def test_manifest_json_redacts_paths():
    df = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "site": "Utrecht",
                "flair_pre_path": "/secret/FLAIR.nii.gz",
                "wmh_path": "/secret/wmh.nii.gz",
            }
        ]
    )
    payload = manifest_json_from_csv(df, root="/secret/root")
    case = payload["cases"][0]
    assert case["modalities"]["flair"]["path"] == "REDACTED_OR_LOCAL_ONLY"
    assert payload["root"] == "REDACTED_OR_LOCAL_ONLY"
    assert "manifest_hash" in payload
