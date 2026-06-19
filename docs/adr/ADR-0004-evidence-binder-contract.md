# ADR-0004: Evidence binder contract

## Status

Accepted

## Context

Preview promotion requires machine-readable evidence linking runs, artifacts, reviews, and blocked claims.

## Decision

- Single register: `registry/release_evidence_register_wmh2017.csv`
- Human index: `docs/release_evidence/`
- Verifiers: `verify_release_evidence_register.py`, `verify_evidence_binder.py`
- CI gate: `evidence_binder_ci.yml` on push/PR

## Consequences

- Structural CI can validate register/docs without local WMH2017 data
- Full artifact hash verification requires local run or `release_candidate_ci` dispatch

## Alternatives

- Manual checklist only (rejected: not auditable at scale)

## Reversal plan

Schema changes require register version bump and contract test updates.
