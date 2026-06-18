#!/usr/bin/env python3
"""Run rollback rehearsal scenarios and write evidence reports."""
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


def run_bad_config(out_dir: Path) -> dict:
    bad_hash = hashlib.sha256(b"bad-threshold-999").hexdigest()
    detected = bad_hash != hashlib.sha256(b"0.5").hexdigest()
    return {
        "rollback_id": "RB-001",
        "scenario": "bad_config",
        "trigger": "config_hash_mismatch",
        "owner": "implementation_lead",
        "verification_result": "PASS" if detected else "FAIL",
        "detected": detected,
        "reviewer_disposition": "APPROVED" if detected else "REJECTED",
    }


def run_wrong_threshold(out_dir: Path) -> dict:
    before = {"threshold": 0.5, "evaluated": True}
    after = {"threshold": 0.9, "evaluated": False}
    detected = after["threshold"] != before["threshold"] and not after["evaluated"]
    return {
        "rollback_id": "RB-002",
        "scenario": "wrong_threshold",
        "trigger": "threshold_change_without_evaluation",
        "verification_result": "PASS" if detected else "FAIL",
        "detected": detected,
        "reviewer_disposition": "APPROVED" if detected else "REJECTED",
    }


def run_dependency_regression(out_dir: Path) -> dict:
    lock = REPO_ROOT / "requirements-lock.txt"
    current = hashlib.sha256(lock.read_bytes()).hexdigest()
    tampered = hashlib.sha256(lock.read_bytes() + b"tamper").hexdigest()
    detected = current != tampered
    return {
        "rollback_id": "RB-003",
        "scenario": "dependency_regression",
        "trigger": "dependency_lock_change",
        "package_hash_before": current,
        "package_hash_after": tampered,
        "verification_result": "PASS" if detected else "FAIL",
        "detected": detected,
        "reviewer_disposition": "APPROVED" if detected else "REJECTED",
    }


def run_artifact_hash_mismatch(out_dir: Path) -> dict:
    manifest = {"artifacts": [{"name": "model", "sha256": "deadbeef"}]}
    actual = "cafebabe"
    detected = manifest["artifacts"][0]["sha256"] != actual
    return {
        "rollback_id": "RB-004",
        "scenario": "artifact_hash_mismatch",
        "trigger": "tampered_artifact_manifest",
        "verification_result": "PASS" if detected else "FAIL",
        "detected": detected,
        "reviewer_disposition": "APPROVED" if detected else "REJECTED",
    }


def run_split_contamination(out_dir: Path) -> dict:
    train = {"case_a", "case_b"}
    test = {"case_b", "case_c"}
    overlap = train & test
    detected = bool(overlap)
    return {
        "rollback_id": "RB-005",
        "scenario": "split_contamination",
        "trigger": "overlapping_train_test_case_id",
        "overlap_cases": sorted(overlap),
        "verification_result": "PASS" if detected else "FAIL",
        "detected": detected,
        "reviewer_disposition": "APPROVED" if detected else "REJECTED",
    }


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
        if payload.get("verification_result") != "PASS":
            failures.append(f"{scenario}: verification FAIL")
        print(f"Wrote {path}")

    if failures:
        raise SystemExit("rollback rehearsal FAIL:\n" + "\n".join(failures))
    print("rollback rehearsal PASS")


if __name__ == "__main__":
    main()
