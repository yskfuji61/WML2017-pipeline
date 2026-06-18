"""Release state determination for WMH2017 offline research package."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Evidence:
    has_committed_secret: bool = False
    has_executable_run_command: bool = True
    has_dependency_lock: bool = True
    has_data_model_code_config_run_linkage: bool = False
    metric_table_regenerable_from_predictions: bool = False
    test_set_isolation_auditable: bool = True
    ci_artifact_hash_recorded: bool = False
    real_data_run_evidence: bool = False
    reviewer_approval: bool = False
    production_claim: bool = False
    monitoring_and_rollback: bool = False


def determine_release_state(evidence: Evidence) -> str:
    if evidence.has_committed_secret:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.has_executable_run_command:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.has_dependency_lock:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.has_data_model_code_config_run_linkage:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.metric_table_regenerable_from_predictions:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.test_set_isolation_auditable:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if not evidence.ci_artifact_hash_recorded:
        return "READY_FOR_STRUCTURAL_REVIEW"
    if not evidence.real_data_run_evidence:
        return "READY_FOR_STRUCTURAL_REVIEW"
    if not evidence.reviewer_approval:
        return "READY_FOR_PREVIEW"
    if evidence.production_claim and not evidence.monitoring_and_rollback:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    return "READY_FOR_PREVIEW"
