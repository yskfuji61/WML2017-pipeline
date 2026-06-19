# Audit gap register

Tracking gaps from the 78-point audit and Wave 1–2 remediation. Status values: `OPEN`, `CLOSED`.

| gap_id | severity | area | description | status | wave | evidence |
|--------|----------|------|-------------|--------|------|----------|
| GAP-001 | Sev1 | docs | Human-facing docs stale vs machine-readable binder/review/decision | CLOSED | 1 | README, release_decision_record, binder index updated |
| GAP-002 | Sev1 | evidence | `release_evidence_register_wmh2017.csv` missing | CLOSED | 1 | registry/release_evidence_register_wmh2017.csv |
| GAP-003 | Sev1 | evidence | `docs/release_evidence/` index missing | CLOSED | 1 | docs/release_evidence/* |
| GAP-004 | Sev1 | supply-chain | Official evaluator not commit/hash/license pinned | OPEN | 1 | pin YAML + fail-closed fetch; fetch still PENDING |
| GAP-005 | Sev1 | security | Missing scan reports treated as PASS | CLOSED | 1 | scan_report.py + Makefile markers |
| GAP-006 | Sev2 | CI | No PR gate for release evidence register structure | CLOSED | 1 | evidence_binder_ci.yml |
| GAP-007 | Sev2 | CI | GitHub Actions green CI URLs not recorded | CLOSED | 3 | scripts/release/update_latest_green_ci.py + structural_ci step |
| GAP-008 | Sev2 | CI | Actions not SHA-pinned | CLOSED | 2 | all workflows + architecture test |
| GAP-009 | Sev2 | architecture | E2E orchestrator monolithic | CLOSED | 2 | src/wmh2017/e2e/ stage split |
| GAP-010 | Sev2 | schema | Full JSON Schema contract for all artifacts | CLOSED | 2 | registry/schemas/* + contract tests |
| GAP-011 | Sev2 | ML risk | No dedicated docs/ml_risk/ reports | CLOSED | 2 | docs/ml_risk/* (7 reports) |
| GAP-012 | Sev2 | ADR | No docs/adr/ decision records | CLOSED | 2 | docs/adr/ ADR-0001–0006 |
| GAP-013 | Sev1 | evaluator | LICENSE_REVIEW disposition NOT_REVIEWED | OPEN | 1 | human review required before fetch |
| GAP-014 | Sev2 | artifacts | Real E2E artifacts gitignored; CI cannot hash-verify locally | OPEN | 1 | FIX-5: CI path-leak gate + derived artifact policy; register hashes; local artifacts remain gitignored |

## Training-start independence (OPEN gaps)

The following OPEN gaps are **not blockers for starting local WMH2017 training runs** (smoke, short full, or full e2e). They remain required before **release-grade** claims (official parity, evaluator fetch, CI hash-verification of local artifacts):

| gap_id | training-start impact | release-claim impact |
|--------|----------------------|----------------------|
| GAP-004 | Independent — official evaluator pin/fetch not needed for MONAI train/eval | Required for official benchmark parity |
| GAP-013 | Independent — local train/eval does not fetch official evaluator | Required before evaluator fetch |
| GAP-014 | Independent — local artifacts stay gitignored by design; hashes recorded in run_manifest after local runs | Required for CI-side artifact hash verification |

Status for these gaps remains **OPEN** until their respective release evidence is completed.
