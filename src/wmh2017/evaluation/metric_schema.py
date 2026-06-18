"""Case metrics schema for WMH2017 local evaluation."""
from __future__ import annotations

REQUIRED_CASE_METRIC_COLUMNS = frozenset(
    {
        "run_id",
        "case_id",
        "assigned_split",
        "site_id",
        "prediction_path",
        "prediction_sha256",
        "label_path",
        "label_sha256",
        "threshold",
        "dice",
        "hd95",
        "avd_percent",
        "lesion_recall",
        "lesion_f1",
        "data_manifest_hash",
        "split_manifest_hash",
        "model_artifact_hash",
        "metric_script_hash",
        "config_hash",
        "code_commit",
        "created_at_utc",
    }
)


def validate_case_metrics_columns(columns: list[str] | set[str]) -> list[str]:
    missing = sorted(REQUIRED_CASE_METRIC_COLUMNS - set(columns))
    return missing
