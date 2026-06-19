# Current state snapshot

Captured for audit traceability after Wave 1 (READY_FOR_PREVIEW P0) implementation.

| Field | Value |
|-------|-------|
| snapshot_date_utc | 2026-06-19 |
| git_commit (pre-wave1-push) | `9a6538e33c3b4fa3e0af4ec66b40f85b26b5dae5` |
| package_version | `0.2.3` |
| local_planning_state | `READY_FOR_STRUCTURAL_REVIEW` |
| controlled_release_state | `READY_FOR_PREVIEW` (structural package; not clinical/production) |
| evidence_binder_status | CLOSED |
| preview_run_id | `wmh2017_preview_20260618_e48ed25` |
| e2e_code_commit | `1d34fd902fe817072e971fcad01fd9695a64d7c9` |
| release_decision | APPROVED_FOR_PREVIEW |

## Blocked claims (unchanged)

Clinical use, customer presentation, proprietary-data processing, unapproved cloud upload, production deployment, SOTA / leaderboard equivalence.

## Machine-readable sources

- `registry/evidence_binder_wmh2017.yaml`
- `registry/review_approval_register_wmh2017.csv`
- `registry/release_evidence_register_wmh2017.csv`
- `docs/release_decisions/release_decision_wmh2017_preview_20260618_e48ed25.yaml`

## Verification (Wave 1)

| command | status | notes |
|---------|--------|-------|
| `make lint` | PASS | ruff clean |
| `make test` | PASS | 86 tests |
| `make typecheck` | PASS | 60 files |
| `make security` | PASS | fail-closed; pip-audit vulns covered by exception register |
| `verify_release_evidence_register.py` | PASS | local artifact hash verification |
| `verify_evidence_binder.py --structure-only` | PASS | CI-safe register/docs gate |
| `verify_evidence_binder.py` (local) | PASS | with `--skip-delegated`; run artifacts present locally |
| `verify_official_evaluator_source.py` | EXPECTED_FAIL | pin commit=PENDING; LICENSE_REVIEW NOT_REVIEWED |

## Open gaps (see audit_gap_register.md)

- GAP-004 / GAP-013: official evaluator not fetched; license not reviewed
- GAP-007: GitHub Actions green CI URLs still PENDING in register
