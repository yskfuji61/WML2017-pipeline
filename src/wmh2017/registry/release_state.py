"""Release state determination for WMH2017 offline research package."""

from __future__ import annotations

from dataclasses import dataclass

RELEASE_LADDER = (
    "STRUCTURAL_REVIEW",
    "PREVIEW_CANDIDATE",
    "READY_FOR_PREVIEW",
    "LIMITED_INTERNAL_USE",
    "READY_FOR_RELEASE",
)


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
    security_gate_pass: bool = False
    reviewer_assigned: bool = False
    reviewer_approval: bool = False
    rollback_rehearsal_complete: bool = False
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
        return "STRUCTURAL_REVIEW"
    if not evidence.real_data_run_evidence:
        return "STRUCTURAL_REVIEW"
    if not evidence.security_gate_pass:
        return "PREVIEW_CANDIDATE"
    if not evidence.reviewer_assigned:
        return "PREVIEW_CANDIDATE"
    if not evidence.reviewer_approval:
        return "PREVIEW_CANDIDATE"
    if evidence.production_claim and not evidence.monitoring_and_rollback:
        return "BLOCKED_BY_SEV0_OR_SEV1"
    if evidence.rollback_rehearsal_complete and evidence.reviewer_approval:
        if evidence.production_claim and evidence.monitoring_and_rollback:
            return "READY_FOR_RELEASE"
        return "READY_FOR_PREVIEW"
    return "READY_FOR_PREVIEW"
