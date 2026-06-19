# Release state policy

## Recommended state enum (v4)

| State | Meaning |
|---|---|
| `READY_FOR_STRUCTURAL_REVIEW` | Static structure review only; no real run claim |
| `REAL_SMOKE_RUN_EVIDENCE_AVAILABLE` | Tiny local WMH2017 smoke run succeeded |
| `STRUCTURAL_REVIEW_HARDENED` | CI/security/reporting hardened |
| `REAL_WMH2017_SMOKE_RUN_EVIDENCE_AVAILABLE` | Full minimal WMH2017 smoke evidence exists |
| `READY_FOR_PREVIEW` | Evidence binder + security + CI + reviewer record exist |
| `NOT_READY_FOR_PREVIEW` | Missing required evidence or open Sev0/Sev1 |
| `NOT_READY_FOR_RELEASE` | Explicit non-claim boundary |

## Prohibited states (positive claims)

- `READY_FOR_RELEASE`
- `CLINICALLY_VALIDATED`
- `CUSTOMER_READY`
- `PRODUCTION_READY`
- `SOTA`

## Repository ceiling

This repository is a public-data local PoC. `determine_release_state()` caps at `READY_FOR_PREVIEW`.
`READY_FOR_RELEASE` must never be claimed.

## Historical evidence

Baseline preview run `wmh2017_preview_20260618_e48ed25` remains valid historical evidence.
New v4 tiny smoke runs are additive and do not invalidate prior preview artifacts.
