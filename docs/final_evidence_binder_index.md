# Final evidence binder index

## Purpose

This binder index lists the exact evidence required before any claim beyond
requirements review. It deliberately separates scaffold evidence from generated
run evidence.

## Binder state

Machine-readable binder: [`registry/evidence_binder_wmh2017.yaml`](../registry/evidence_binder_wmh2017.yaml)

Verifier: `python scripts/verify_evidence_binder.py --run-id <run_id> --target-state READY_FOR_PREVIEW`

```text
binder_status: OPEN
real_wmh2017_run_evidence: GENERATED_LOCALLY_NOT_COMMITTED
official_metric_parity: NOT_REVIEWED
source_license_review: NOT_COMPLETED
review_approval: NOT_COMPLETED
release_decision: NOT_APPROVED
```

## Required binder sections

| Section | Required artifact | Current status | Blocking severity |
|---|---|---:|---|
| Package identity | `reports/full_package_manifest.json` | generated for refactored ZIP | Sev2 until reviewer checks it |
| Source register | `registry/source_register_wmh2017.csv` | present but review incomplete | Sev1 |
| Claim boundary | `registry/claim_boundary_wmh2017.csv` | present | Sev2 until reviewed |
| Dataset manifest | `artifacts/runs/<run_id>/dataset/dataset_manifest.json` | generated locally; not committed | Sev1 |
| Label audit | `artifacts/runs/<run_id>/label_audit/label_audit.json` | generated locally | Sev1 |
| Split manifest | `artifacts/runs/<run_id>/splits/split_manifest.json` | generated locally | Sev1 |
| Overlay evidence | `reports/overlays/*_overlay.png` + reviewer note | not generated | Sev2 |
| Training log | `artifacts/runs/<run_id>/logs/train_log.jsonl` | not generated | Sev1 |
| Run evidence | `artifacts/runs/<run_id>/run_context.json` + `artifact_manifest.json` | generated locally | Sev1 |
| Checkpoint hash | `artifacts/runs/<run_id>/checkpoints/*` + sha256 | not generated | Sev1 |
| Prediction artifacts | `artifacts/runs/<run_id>/predictions/*_pred.nii.gz` + sha256 | not generated | Sev1 |
| Local metric output | `case_metrics.csv`, `metrics_summary.json` | not generated | Sev1 |
| Official evaluator parity | official evaluator hash + parity note | not completed | Sev2, Sev1 for leaderboard claim |
| Finding register | `registry/finding_register_wmh2017.csv` | created with open findings | Sev1 open |
| Review record | `registry/review_approval_register_wmh2017.csv` | created, not completed | Sev1 |
| Release decision | `docs/release_decision_record.md` | updated as non-approval | Sev1 for release claim |
| Rollback plan | `docs/rollback_plan_wmh2017.md` | created | Sev2 until exercised |
| Inspection retrieval | `docs/inspection_retrieval_plan.md` | created | Sev2 until owner/SLA confirmed |

## Binder closure rule

The binder cannot be closed until every Sev1 is either closed with evidence or
the release claim is reduced so the finding is no longer release-critical.
