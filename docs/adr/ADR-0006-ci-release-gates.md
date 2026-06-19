# ADR-0006: CI release gates

## Status

Accepted

## Context

Multiple CI workflows must enforce structural, security, supply-chain, and evidence gates before preview promotion.

## Decision

| Workflow | Gate |
|----------|------|
| structural-ci | lint, format, typecheck, tests, architecture scripts |
| security-scan | bandit, pip-audit, detect-secrets, fail-closed policy |
| evidence-binder-ci | register + binder structure |
| dependency-review | PR dependency severity (fail on high) |
| license-scan | SBOM + license_report.json non-empty |
| release-candidate-ci | full E2E + lineage + evidence (manual dispatch) |

All workflows pin Actions to full commit SHAs.

## Consequences

- Green structural/security/evidence CI required for routine merges
- Full E2E hash verification remains manual/dispatch (GAP-014 OPEN)

## Alternatives

- Single monolithic workflow (rejected: slow feedback on PRs)

## Reversal plan

Gate changes require ADR update and `latest_green_ci.md` workflow table sync.
