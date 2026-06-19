# Audit gap register

Tracking gaps from the 78-point audit and Wave 1 remediation. Status values: `OPEN`, `CLOSED`, `WAVE2`.

| gap_id | severity | area | description | status | wave | evidence |
|--------|----------|------|-------------|--------|------|----------|
| GAP-001 | Sev1 | docs | Human-facing docs stale vs machine-readable binder/review/decision | CLOSED | 1 | README, release_decision_record, binder index updated |
| GAP-002 | Sev1 | evidence | `release_evidence_register_wmh2017.csv` missing | CLOSED | 1 | registry/release_evidence_register_wmh2017.csv |
| GAP-003 | Sev1 | evidence | `docs/release_evidence/` index missing | CLOSED | 1 | docs/release_evidence/* |
| GAP-004 | Sev1 | supply-chain | Official evaluator not commit/hash/license pinned | OPEN | 1 | pin YAML + fail-closed fetch; fetch still PENDING |
| GAP-005 | Sev1 | security | Missing scan reports treated as PASS | CLOSED | 1 | scan_report.py + Makefile markers |
| GAP-006 | Sev2 | CI | No PR gate for release evidence register structure | CLOSED | 1 | evidence_binder_ci.yml |
| GAP-007 | Sev2 | CI | GitHub Actions green CI URLs not recorded | OPEN | 1 | latest_green_ci.md uses PENDING until CI run |
| GAP-008 | Sev2 | CI | Actions not SHA-pinned | WAVE2 | 2 | deferred |
| GAP-009 | Sev2 | architecture | E2E orchestrator monolithic | WAVE2 | 2 | deferred stage split |
| GAP-010 | Sev2 | schema | Full JSON Schema contract for all artifacts | WAVE2 | 2 | release_evidence only in Wave 1 |
| GAP-011 | Sev2 | ML risk | No dedicated docs/ml_risk/ reports | WAVE2 | 2 | deferred |
| GAP-012 | Sev2 | ADR | No docs/adr/ decision records | WAVE2 | 2 | deferred |
| GAP-013 | Sev1 | evaluator | LICENSE_REVIEW disposition NOT_REVIEWED | OPEN | 1 | human review required before fetch |
| GAP-014 | Sev2 | artifacts | Real E2E artifacts gitignored; CI cannot hash-verify locally | OPEN | 1 | register records hashes; release_candidate_ci with WMH2017_ROOT |
