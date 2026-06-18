import json
from pathlib import Path

import pandas as pd
import pytest

from wmh2017.evaluation.official_parity import ParityConfig, compare_official_parity


def test_official_parity_passes_with_alias_columns(tmp_path):
    local = tmp_path / "local.csv"
    official = tmp_path / "official.csv"
    pd.DataFrame(
        [
            {"case_id": "A", "dice": 0.9, "hd95": 1.0, "avd_percent": 2.0},
            {"case_id": "B", "dice": 0.8, "hd95": 2.0, "avd_percent": 3.0},
        ]
    ).to_csv(local, index=False)
    pd.DataFrame(
        [
            {"subject_id": "A", "DSC": 0.9, "H95": 1.0, "AVD": 2.0},
            {"subject_id": "B", "DSC": 0.8, "H95": 2.0, "AVD": 3.0},
        ]
    ).to_csv(official, index=False)

    report = compare_official_parity(
        local,
        official,
        tmp_path / "out",
        config=ParityConfig(required_metrics=("dice", "hd95", "avd_percent")),
    )

    assert report["status"] == "passed"
    assert report["n_compared_cases"] == 2
    assert Path(report["case_diff_csv"]).exists()
    saved = json.loads((tmp_path / "out" / "official_parity_report.json").read_text(encoding="utf-8"))
    assert saved["metric_summaries"]["dice"]["all_within_tolerance"] is True


def test_official_parity_fails_on_metric_difference(tmp_path):
    local = tmp_path / "local.csv"
    official = tmp_path / "official.csv"
    pd.DataFrame([{"case_id": "A", "dice": 0.9}]).to_csv(local, index=False)
    pd.DataFrame([{"case_id": "A", "Dice": 0.7}]).to_csv(official, index=False)

    report = compare_official_parity(
        local,
        official,
        tmp_path / "out",
        config=ParityConfig(required_metrics=("dice",), tolerances={"dice": 1e-6}),
    )

    assert report["status"] == "failed"
    assert report["metric_summaries"]["dice"]["all_within_tolerance"] is False


def test_official_parity_rejects_case_mismatch(tmp_path):
    local = tmp_path / "local.csv"
    official = tmp_path / "official.csv"
    pd.DataFrame([{"case_id": "A", "dice": 0.9}]).to_csv(local, index=False)
    pd.DataFrame([{"case_id": "B", "Dice": 0.9}]).to_csv(official, index=False)

    with pytest.raises(ValueError, match="case mismatch"):
        compare_official_parity(
            local,
            official,
            tmp_path / "out",
            config=ParityConfig(required_metrics=("dice",)),
        )
