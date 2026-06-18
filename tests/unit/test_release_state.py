from wmh2017.registry.release_state import Evidence, determine_release_state


def test_default_evidence_is_structural_review():
    assert determine_release_state(Evidence()) == "BLOCKED_BY_SEV0_OR_SEV1"


def test_structural_review_when_ci_only():
    evidence = Evidence(
        has_data_model_code_config_run_linkage=True,
        metric_table_regenerable_from_predictions=True,
    )
    assert determine_release_state(evidence) == "READY_FOR_STRUCTURAL_REVIEW"


def test_preview_when_reviewer_missing():
    evidence = Evidence(
        has_data_model_code_config_run_linkage=True,
        metric_table_regenerable_from_predictions=True,
        ci_artifact_hash_recorded=True,
        real_data_run_evidence=True,
    )
    assert determine_release_state(evidence) == "READY_FOR_PREVIEW"
