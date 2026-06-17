# Rollback plan — WMH2017 local PoC scaffold

## Current rollback class

```text
rollback_class: local_repository_artifacts_only
raw_data_action: never delete raw dataset automatically
cloud_action: none allowed
customer_action: none allowed
```

## Rollback triggers

- protected information appears in repository, logs, reports, or outputs
- source/license review rejects current use
- test data contamination enters train/validation/tuning
- label==2 is used as foreground
- manifest/hash mismatch after approval
- official metric parity fails after a comparison claim
- reviewer rejects claim boundary
- any Sev0/Sev1 is discovered after promotion

## Restore steps

1. Stop new runs and external sharing.
2. Preserve current package and run evidence as incident evidence.
3. Record trigger in `registry/finding_register_wmh2017.csv`.
4. Revert code/config/registry files to the last accepted package hash.
5. Delete generated non-record artifacts only after owner approval.
6. Never delete raw public/proprietary data automatically.
7. Re-run unit tests and affected evidence checks.
8. Update release decision record and notify owner/reviewer.

## Required rollback evidence

- trigger ID
- owner
- package hash before rollback
- package hash after rollback
- affected files
- affected claims
- affected runs
- evidence preserved
- verification commands
- reviewer disposition
