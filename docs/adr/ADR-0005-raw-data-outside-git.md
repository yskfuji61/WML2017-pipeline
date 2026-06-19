# ADR-0005: Raw data outside git

## Status

Accepted

## Context

WMH2017 imaging data must not be committed. Operators provide `WMH2017_ROOT` locally or via CI secret.

## Decision

- Raw data paths gitignored; `verify_no_raw_data_committed.py` in CI
- Dataset audit stages validate structure against local root only
- Artifact hashes recorded in release evidence register for preview runs

## Consequences

- Hosted PR CI cannot run full real-data E2E without secrets
- `release_candidate_ci` uses `WMH2017_ROOT` secret and `preview-candidate` environment

## Alternatives

- Git LFS for subset (rejected: license and size constraints)

## Reversal plan

If data policy changes, update ADR, `.gitignore`, and audit gates before any commit of raw data.
