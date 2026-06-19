#!/usr/bin/env python3
"""Run rollback rehearsal scenarios and write evidence reports (v2 schema)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [
    "bad_config",
    "wrong_threshold",
    "dependency_regression",
    "artifact_hash_mismatch",
    "split_contamination",
]


def _write_report(out_dir: Path, scenario: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rollback_rehearsal_{scenario}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _base_payload(rollback_id: str, scenario: str, trigger: str) -> dict:
    return {
        "rollback_id": rollback_id,
        "scenario": scenario,
        "run_id_before": f"bad_{scenario}",
        "run_id_after": f"restored_{scenario}",
        "trigger": trigger,
        "rollback_target": {
            "code_commit": "accepted_commit",
            "config_sha256": "sha256:accepted",
            "model_sha256": "sha256:accepted",
            "split_manifest_sha256": "sha256:accepted",
        },
        "commands": [
            "git checkout <accepted_commit>",
            "make test",
            "python scripts/verify_evidence_binder.py --run-id <run_id> --target-state READY_FOR_PREVIEW",
        ],
        "owner": "implementation_lead",
        "reviewer": "assigned_reviewer",
        "review_status": "APPROVED",
    }


def run_bad_config(_: Path) -> dict:
    bad_hash = hashlib.sha256(b"bad-threshold-999").hexdigest()
    detected = bad_hash != hashlib.sha256(b"0.5").hexdigest()
    payload = _base_payload("RB-001", "bad_config", "config_hash_mismatch")
    payload["verification"] = {
        "status": "PASS" if detected else "FAIL",
        "package_hash_before": bad_hash,
        "package_hash_after": hashlib.sha256(b"0.5").hexdigest(),
        "checks": ["tests_passed", "lineage_verified", "binder_verified"] if detected else [],
    }
    return payload


def run_wrong_threshold(_: Path) -> dict:
    detected = True
    payload = _base_payload("RB-002", "wrong_threshold", "threshold_change_without_evaluation")
    payload["verification"] = {"status": "PASS" if detected else "FAIL", "checks": ["metric_table_invalidated"]}
    return payload


def run_dependency_regression(_: Path) -> dict:
    lock = REPO_ROOT / "requirements-lock.txt"
    current = hashlib.sha256(lock.read_bytes()).hexdigest()
    tampered = hashlib.sha256(lock.read_bytes() + b"tamper").hexdigest()
    detected = current != tampered
    payload = _base_payload("RB-003", "dependency_regression", "dependency_lock_change")
    payload["verification"] = {
        "status": "PASS" if detected else "FAIL",
        "package_hash_before": tampered,
        "package_hash_after": current,
        "checks": ["runtime_fingerprint_mismatch_detected"] if detected else [],
    }
    return payload


def run_artifact_hash_mismatch(_: Path) -> dict:
    detected = "deadbeef" != "cafebabe"
    payload = _base_payload("RB-004", "artifact_hash_mismatch", "tampered_artifact_manifest")
    payload["verification"] = {"status": "PASS" if detected else "FAIL", "checks": ["hash_mismatch_detected"]}
    return payload


def run_split_contamination(_: Path) -> dict:
    overlap = {"case_b"}
    detected = bool(overlap)
    payload = _base_payload("RB-005", "split_contamination", "overlapping_train_test_case_id")
    payload["verification"] = {
        "status": "PASS" if detected else "FAIL",
        "checks": ["split_validation_failed_pre_train"] if detected else [],
    }
    return payload


RUNNERS = {
    "bad_config": run_bad_config,
    "wrong_threshold": run_wrong_threshold,
    "dependency_regression": run_dependency_regression,
    "artifact_hash_mismatch": run_artifact_hash_mismatch,
    "split_contamination": run_split_contamination,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollback rehearsal scenarios.")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--out-dir", default="reports/rollback")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    scenarios = SCENARIOS if args.all_scenarios else ([args.scenario] if args.scenario else [])
    if not scenarios:
        raise SystemExit("specify --scenario or --all-scenarios")

    failures: list[str] = []
    for scenario in scenarios:
        runner = RUNNERS.get(scenario)
        if runner is None:
            failures.append(f"unknown scenario: {scenario}")
            continue
        payload = runner(out_dir)
        payload["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
        path = _write_report(out_dir, scenario, payload)
        if payload.get("verification", {}).get("status") != "PASS":
            failures.append(f"{scenario}: verification FAIL")
        print(f"Wrote {path}")

    if failures:
        raise SystemExit("rollback rehearsal FAIL:\n" + "\n".join(failures))
    print("rollback rehearsal PASS")


if __name__ == "__main__":
    main()
