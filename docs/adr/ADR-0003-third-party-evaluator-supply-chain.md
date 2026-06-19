# ADR-0003: Third-party evaluator supply chain

## Status

Accepted

## Context

Official WMH challenge metrics depend on an external evaluator repository. Unpinned fetch creates supply-chain and license risk.

## Decision

- Pin evaluator source in registry YAML with commit SHA and license review disposition
- `verify_official_evaluator_source.py` fails closed when commit is `PENDING` or license is `NOT_REVIEWED`
- No automatic fetch until human `LICENSE_REVIEW` approval

## Consequences

- `official_comparable` claims blocked (GAP-004/013 remain OPEN until fetch)
- Parity report limited to local fixtures

## Alternatives

- Vendor evaluator into repo without review (rejected: license unknown)

## Reversal plan

After APPROVED license review, fetch pinned commit, update SHA256, re-run parity on fixtures then real cases.
