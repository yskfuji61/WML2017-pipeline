# ADR-0002: Preview-not-release policy

## Status

Accepted

## Context

The project targets `READY_FOR_PREVIEW` as the first controlled release state, not clinical or production release.

## Decision

- Controlled release state: `READY_FOR_PREVIEW`
- Blocked claims documented in README, release decision, and ML risk reports
- `READY_FOR_RELEASE` requires additional human review and gap closure (GAP-004/013)

## Consequences

- Evidence binder and register use preview-oriented wording
- Performance, clinical, and SOTA claims remain forbidden

## Alternatives

- Skip preview state and aim for full release (rejected: audit gaps remain open)

## Reversal plan

Promote to stronger state only after GAP-004/013 closure and explicit release decision update.
