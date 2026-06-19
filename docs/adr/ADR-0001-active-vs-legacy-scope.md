# ADR-0001: Active vs legacy scope

## Status

Accepted

## Context

The repository contains both the active WMH2017 package (`src/wmh2017/`) and legacy ISLES-derived reference code (`core/pipeline/`).

## Decision

- **Active:** `src/wmh2017/**`, root scripts, configs, and WMH2017 registry artifacts
- **Legacy (reference only):** `core/pipeline/**` — not imported by active smoke/E2E path

## Consequences

- CI gates (`verify_no_legacy_imports.py`) enforce no active imports from legacy paths
- Documentation and release evidence refer to `src/wmh2017/` only

## Alternatives

- Delete legacy code (rejected: useful reference for Phase 2 ensemble work)

## Reversal plan

If legacy is promoted to active, create ADR superseding this record and update import gates.
