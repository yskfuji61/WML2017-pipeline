# Current state snapshot

Captured for audit traceability after Wave 2 (READY_FOR_PREVIEW P1) implementation.

| Field | Value |
|-------|-------|
| snapshot_date_utc | 2026-06-19 |
| git_commit (pre-wave2-push) | `a08997f` |
| package_version | `0.2.3` |
| local_planning_state | `READY_FOR_STRUCTURAL_REVIEW` |
| controlled_release_state | `READY_FOR_PREVIEW` (structural package; not clinical/production) |
| evidence_binder_status | CLOSED |
| preview_run_id | `wmh2017_preview_20260618_e48ed25` |
| e2e_code_commit | `1d34fd902fe817072e971fcad01fd9695a64d7c9` |
| release_decision | APPROVED_FOR_PREVIEW |

## Blocked claims (unchanged)

Clinical use, customer presentation, proprietary-data processing, unapproved cloud upload, production deployment, SOTA / leaderboard equivalence.

## Wave 2 deliverables

- E2E stage split: `src/wmh2017/e2e/` + thin `scripts/run_wmh2017_e2e.py`
- CI hardening: Actions SHA pin, `dependency_review.yml`, `license_scan.yml`, ruff format check
- Artifact schemas: `src/wmh2017/registry/schemas/*` + `validate_artifact_schema.py`
- ML risk docs: `docs/ml_risk/` (7 reports)
- ADRs: `docs/adr/` (ADR-0001–0006)
- Observability: `event_log.py`, `failure_summary.py` wired in E2E runner

## Machine-readable sources

- `registry/evidence_binder_wmh2017.yaml`
- `registry/review_approval_register_wmh2017.csv`
- `registry/release_evidence_register_wmh2017.csv`
- `docs/release_decisions/release_decision_wmh2017_preview_20260618_e48ed25.yaml`

## Verification (Wave 2)

| command | status | notes |
|---------|--------|-------|
| `make lint` | PASS | ruff check + format --check |
| `make test` | PASS | 95 tests |
| `make typecheck` | PASS | mypy src scripts |
| `make security` | PASS | fail-closed; pip-audit vulns in exception register |
| `validate_artifact_schema.py` | PASS | preview run artifacts |
| `verify_release_evidence_register.py` | PASS | local artifact hash verification |
| `verify_evidence_binder.py --structure-only` | PASS | CI-safe register/docs gate |
| `verify_official_evaluator_source.py` | EXPECTED_FAIL | pin commit=PENDING; LICENSE_REVIEW NOT_REVIEWED |

## Open gaps (see audit_gap_register.md)

- GAP-004 / GAP-013: official evaluator not fetched; license not reviewed
- GAP-007: GitHub Actions green CI URLs still PENDING in register
- GAP-014: real E2E artifacts gitignored; full hash gate needs dispatch CI or local run
