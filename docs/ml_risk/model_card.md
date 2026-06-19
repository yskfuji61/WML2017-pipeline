# Model card — WMH2017 smoke baseline

## Model

| Field | Value |
|-------|-------|
| architecture | MONAI 3D U-Net (smoke config) |
| training scope | Challenge training cases only (see split policy) |
| checkpoint | `artifacts/runs/<run_id>/checkpoints/` (local only) |

Fragment artifact: `model_card_fragment.json` written per E2E run.

## Intended use

- Research reproducibility and audit trail for WMH2017 public-data PoC
- Structural preview package validation

## Out of scope

- Clinical decision support
- Production deployment
- Customer-facing performance guarantees

## Blocked claims

See [limitations.md](limitations.md). SOTA, clinical, and production claims remain blocked.
