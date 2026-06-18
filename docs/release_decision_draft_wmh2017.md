# Release Decision Draft — WMH2017

## Current decision

```text
READY_FOR_REQUIREMENTS_REVIEW
```

## Not approved for

- customer presentation,
- clinical use,
- diagnostic support,
- production deployment,
- SOTA claim,
- independent cloud processing,
- private/PHI data use.

## Promotion criteria

### PIPELINE_SMOKE_READY

- EXP-000 passes,
- unit tests pass,
- one overlay generated,
- split manifest created,
- run evidence exists.

### BASELINE_READY

- MONAI smoke training completes,
- validation metrics recorded,
- failure notes recorded,
- no test contamination.

### REPRODUCTION_READY

- winner source/license verified,
- reproduction run completed,
- deviations recorded,
- all five metrics computed.

### CLAIM_REVIEW_READY

- comparison validity record exists,
- source/evidence/security/domain reviewers approve,
- Sev0/Sev1 findings closed.

## Rollback

Generated artifacts and registries can be regenerated from:
- code commit,
- config hash,
- dataset manifest hash,
- split manifest hash,
- environment hash.

Raw dataset remains read-only and outside repo.
