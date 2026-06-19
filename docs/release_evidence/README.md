# Release evidence index

Machine-readable register: [`registry/release_evidence_register_wmh2017.csv`](../../registry/release_evidence_register_wmh2017.csv)

Verifier: `python scripts/verify_release_evidence_register.py --run-id wmh2017_preview_20260618_e48ed25`

## Documents

| Document | Purpose |
|----------|---------|
| [latest_green_ci.md](latest_green_ci.md) | Structural/security/evidence CI status |
| [real_wmh2017_e2e_run.md](real_wmh2017_e2e_run.md) | Real-data E2E run_id and artifact hashes |
| [official_evaluator_parity.md](official_evaluator_parity.md) | Official metric parity status |
| [human_review_record.md](human_review_record.md) | Human review approvals |
| [artifact_hash_manifest.md](artifact_hash_manifest.md) | SHA256 manifest for preview run |
| [release_decision.md](release_decision.md) | Release decision summary |

## ML risk reports

| Document | Purpose |
|----------|---------|
| [docs/ml_risk/](../ml_risk/) | Split isolation, preprocessing parity, metric parity, limitations |

## Blocked claims

Clinical use, customer presentation, proprietary-data processing, cloud upload, production deployment, leaderboard/SOTA equivalence remain blocked.
