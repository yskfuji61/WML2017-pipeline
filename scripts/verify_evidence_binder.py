#!/usr/bin/env python3
"""Verify evidence binder closure for target release state (v2 §8.2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from wmh2017.lineage.hashes import verify_sidecar

PREVIEW_STATES = {"READY_FOR_PREVIEW", "READY_FOR_LIMITED_INTERNAL_USE", "READY_FOR_RELEASE"}
BLOCKED_CLAIMS = {
    "clinical_use",
    "customer_presentation",
    "cloud_upload",
    "production_deployment",
    "leaderboard_or_sota_claim",
    "SOTA_or_leaderboard_equivalence",
}


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_script(repo_root: Path, script: str, *extra: str) -> None:
    cmd = [sys.executable, str(repo_root / "scripts" / script), *extra]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _required_for(section: dict, target_state: str) -> bool:
    required_for = section.get("required_for", [])
    if not required_for:
        return bool(section.get("required"))
    return target_state in required_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evidence binder closure.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--target-state", default="READY_FOR_PREVIEW")
    parser.add_argument("--binder", default="registry/evidence_binder_wmh2017.yaml")
    parser.add_argument("--skip-delegated", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    binder_path = repo_root / args.binder
    binder = yaml.safe_load(binder_path.read_text(encoding="utf-8")) or {}
    sections = binder.get("required_sections", {})
    failures: list[str] = []

    # 1. package identity consistency
    if not args.skip_delegated:
        try:
            _run_script(repo_root, "verify_package_identity.py")
        except subprocess.CalledProcessError as exc:
            failures.append(f"check 1 package identity failed: {exc}")

    # 2. artifact existence (package identity section)
    for rel in sections.get("package_identity", {}).get("artifacts", []):
        if not (repo_root / rel).exists():
            failures.append(f"check 2 missing package identity artifact: {rel}")

    # 3. artifact sha256 match (run artifacts with sidecars)
    if _required_for(sections.get("real_run", {}), args.target_state) and args.run_id:
        for pattern in sections.get("real_run", {}).get("artifacts", []):
            rel = pattern.replace("{run_id}", args.run_id)
            path = repo_root / rel
            if not path.exists():
                failures.append(f"check 3 missing run artifact: {rel}")
                continue
            sidecar = path.parent / (path.name + ".sha256")
            if path.suffix in {".json", ".yaml", ".csv", ".pt"} and sidecar.exists():
                if not verify_sidecar(path, sidecar):
                    failures.append(f"check 3 sha256 mismatch: {rel}")

    # 4. run_id consistency
    if args.run_id:
        ctx_path = repo_root / "artifacts" / "runs" / args.run_id / "run_context.json"
        if ctx_path.exists():
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            if ctx.get("run_id") != args.run_id:
                failures.append("check 4 run_id mismatch in run_context")

    # 5. code_commit consistency
    if args.run_id:
        ctx_path = repo_root / "artifacts" / "runs" / args.run_id / "run_context.json"
        graph_path = repo_root / "artifacts" / "runs" / args.run_id / "lineage" / "lineage_graph.json"
        if ctx_path.exists() and graph_path.exists():
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            if ctx.get("code_commit") and graph.get("code_commit") and ctx["code_commit"] != graph["code_commit"]:
                failures.append("check 5 code_commit mismatch between run_context and lineage_graph")

    # 6. full_package_manifest sha256 recorded in release decision
    manifest_path = repo_root / "reports" / "full_package_manifest.json"
    if manifest_path.exists() and args.run_id:
        decision_path = repo_root / "docs/release_decisions" / f"release_decision_{args.run_id}.yaml"
        if decision_path.exists():
            decision = yaml.safe_load(decision_path.read_text(encoding="utf-8")) or {}
            expected = str(decision.get("package_manifest_sha256", "")).replace("sha256:", "")
            if expected and expected not in {"PENDING", "PENDING_GENERATED_MANIFEST"}:
                actual = _sha256_file(manifest_path)
                if actual != expected:
                    failures.append("check 6 package_manifest_sha256 mismatch in release decision")

    # 7. finding register
    if args.target_state in PREVIEW_STATES and not args.skip_delegated:
        try:
            _run_script(repo_root, "verify_finding_register.py", "--target-state", args.target_state)
        except subprocess.CalledProcessError:
            failures.append("check 7 finding register failed")

    # 8. review register approved
    if args.target_state in PREVIEW_STATES and not args.skip_delegated:
        extra = ["--target-state", args.target_state]
        if args.run_id:
            extra.extend(["--run-id", args.run_id])
        try:
            _run_script(repo_root, "verify_review_approval.py", *extra)
        except subprocess.CalledProcessError:
            failures.append("check 8 review register failed")

    # 9. release decision approved
    if args.target_state in PREVIEW_STATES and args.run_id:
        decision_path = repo_root / "docs/release_decisions" / f"release_decision_{args.run_id}.yaml"
        if _required_for(sections.get("release_decision", {}), args.target_state):
            if not decision_path.exists():
                failures.append(f"check 9 missing release decision: {decision_path}")
            else:
                decision = yaml.safe_load(decision_path.read_text(encoding="utf-8")) or {}
                if decision.get("decision", "").startswith("NOT_APPROVED"):
                    failures.append("check 9 release decision not approved")

    # 10. security policy PASS
    if _required_for(sections.get("security", {}), args.target_state):
        policy = repo_root / "reports/security/security_policy_result.json"
        if not policy.exists():
            failures.append("check 10 missing security_policy_result.json")
        else:
            result = json.loads(policy.read_text(encoding="utf-8"))
            if result.get("status") != "PASS":
                failures.append("check 10 security_policy_result not PASS")

    # 11. SBOM non-empty
    if _required_for(sections.get("security", {}), args.target_state):
        sbom_path = repo_root / "reports/security/sbom.cdx.json"
        if not sbom_path.exists():
            failures.append("check 11 SBOM missing")
        else:
            data = json.loads(sbom_path.read_text(encoding="utf-8"))
            if not data.get("components"):
                failures.append("check 11 SBOM empty")

    # 12. rollback (target-state dependent)
    if _required_for(sections.get("rollback", {}), args.target_state):
        for rel in sections.get("rollback", {}).get("artifacts", []):
            if not (repo_root / rel).exists():
                failures.append(f"check 12 missing rollback artifact: {rel}")

    # 13. parity vs claim boundary
    parity_path = repo_root / "reports/evaluation/official_metric_parity_report.json"
    if _required_for(sections.get("official_metric_parity", {}), args.target_state):
        if not parity_path.exists():
            failures.append("check 13 missing official_metric_parity_report.json")
        else:
            report = json.loads(parity_path.read_text(encoding="utf-8"))
            claims = report.get("claims_allowed", report.get("claim_allowed", {}))
            if claims.get("leaderboard_or_sota"):
                failures.append("check 13 blocked SOTA claim allowed in parity report")

    # 14. blocked claims not approved
    if args.run_id:
        decision_path = repo_root / "docs/release_decisions" / f"release_decision_{args.run_id}.yaml"
        if decision_path.exists():
            decision = yaml.safe_load(decision_path.read_text(encoding="utf-8")) or {}
            approved = set(decision.get("claims_approved", []))
            if approved & BLOCKED_CLAIMS:
                failures.append(f"check 14 blocked claims accidentally approved: {approved & BLOCKED_CLAIMS}")

    # lineage section existence
    if _required_for(sections.get("lineage", {}), args.target_state) and args.run_id:
        lineage_rel = f"artifacts/runs/{args.run_id}/lineage/lineage_graph.json"
        if not (repo_root / lineage_rel).exists():
            failures.append(f"missing lineage artifact: {lineage_rel}")

    if failures:
        raise SystemExit("evidence binder gate FAIL:\n" + "\n".join(failures))
    print(f"evidence binder gate PASS for target {args.target_state}")


if __name__ == "__main__":
    main()
