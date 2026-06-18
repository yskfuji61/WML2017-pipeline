"""Case metrics schema for WMH2017 local evaluation."""
from __future__ import annotations

REQUIRED_CASE_METRIC_COLUMNS = frozenset(
    {
        "run_id",
        "case_id",
        "assigned_split",
        "site_or_center",
        "label_path_redacted",
        "label_sha256",
        "prediction_path_redacted",
        "prediction_sha256",
        "model_artifact_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "metric_script_sha256",
        "threshold",
        "dice",
        "hd95",
        "avd",
        "lesion_recall",
        "lesion_f1",
        "created_at_utc",
        "code_commit",
    }
)


def validate_case_metrics_columns(columns: list[str] | set[str]) -> list[str]:
    missing = sorted(REQUIRED_CASE_METRIC_COLUMNS - set(columns))
    return missing
