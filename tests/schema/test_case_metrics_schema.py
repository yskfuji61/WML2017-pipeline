from wmh2017.evaluation.metric_schema import REQUIRED_CASE_METRIC_COLUMNS, validate_case_metrics_columns


def test_required_case_metric_columns_complete():
    assert "site_id" in REQUIRED_CASE_METRIC_COLUMNS
    assert "model_artifact_hash" in REQUIRED_CASE_METRIC_COLUMNS
    assert "config_hash" in REQUIRED_CASE_METRIC_COLUMNS
    assert "created_at_utc" in REQUIRED_CASE_METRIC_COLUMNS


def test_validate_case_metrics_columns_detects_missing():
    missing = validate_case_metrics_columns(["run_id", "case_id", "dice"])
    assert "site_id" in missing
    assert "model_artifact_hash" in missing
