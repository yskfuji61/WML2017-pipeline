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
- GAP-007: GitHub Actions green CI URLs recorded at `docs/release_evidence/latest_green_ci.md` (commit `d37e613`)
- GAP-014: real E2E artifacts gitignored; full hash gate needs dispatch CI or local run

## 2026-06-22 update (post Phase B2 CV)

| Field | Value |
|-------|-------|
| update_date_utc | 2026-06-22 |
| git_commit | `28662b0` |
| change | Phase A selection semantics (ADR-0007), Phase B1 k-fold foundation, Phase B2 5-fold CV |
| cv_id | `wmh2017_a2cv_cosine_seed42` (summary: `reports/cv/cv_summary_a2cv_cosine_seed42.json`) |
| cv_result | mean_dice 0.614 +/- 0.037, lesion_recall 0.207 +/- 0.038, lesion_f1 0.297 +/- 0.047 (n=5) |
| gate | Phase A (0.65/0.35) NOT met; Phase B (0.72) NOT met |

Blocked claims unchanged (clinical/customer/proprietary/cloud/production/SOTA).
All metrics are local validation only; the test split is never used.

## 2026-06-22 update (RC2 recall redesign CV)

| Field | Value |
|-------|-------|
| update_date_utc | 2026-06-22 |
| git_commit | `d37e613` |
| change | RC2 recall redesign selected after fold0 probe; 5-fold CV completed |
| cv_id | `wmh2017_rc2_seed42` (summary: `reports/cv/cv_summary_rc2_seed42.json`) |
| cv_result | mean_dice 0.612 +/- 0.047, lesion_recall 0.272 +/- 0.084, lesion_f1 0.354 +/- 0.056 (n=5) |
| fold0_reference | dice 0.671, recall 0.252 (`wmh2017_rc2_cosine_fold0_seed42`) |
| vs_a2cv | recall +0.065; dice -0.002 |
| gate | Phase A (0.65/0.35) **NOT met**; Phase B (0.72) **NOT met** |
| gate_evidence | `docs/release_evidence/phase_gate_judgment_rc2.md` |

Prior A2-CV row above remains historical; RC2 is the current primary CV measurement.
