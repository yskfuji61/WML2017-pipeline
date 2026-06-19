# Preview promotion plan (READY_FOR_PREVIEW)

Goal: promote the structural research PoC to `READY_FOR_PREVIEW` without clinical, customer, production, or SOTA claims.

## Preconditions

- All Sev1 findings in `registry/finding_register_wmh2017.csv` are CLOSED for preview target.
- Four human reviews APPROVED in `registry/review_approval_register_wmh2017.csv`.
- Raw WMH2017 data remains outside git.

## Promotion steps

1. Install locked environment: `make setup`
2. Run static gates: `make lint`, `make typecheck`, `make test`
3. Run real-data E2E:
   ```bash
   export WMH2017_ROOT=/path/to/MICCAI2017_WMH/files
   make preview-candidate RUN_ID=wmh2017_preview_YYYYMMDD_gitsha WMH2017_ROOT="$WMH2017_ROOT"
   ```
4. Update `registry/release_evidence_register_wmh2017.csv` with run_id, commit_sha, artifact hashes.
5. Update `docs/release_evidence/*` human summaries.
6. Record release decision YAML under `docs/release_decisions/`.
7. Verify:
   ```bash
   python scripts/verify_release_evidence_register.py --run-id "$RUN_ID"
   python scripts/verify_evidence_binder.py --run-id "$RUN_ID" --target-state READY_FOR_PREVIEW
   python scripts/verify_lineage_graph.py --run-id "$RUN_ID" --require-artifact-hashes
   ```
8. Human release owner updates README controlled state (no blocked-claim relaxation).

## Blocked after promotion

- `READY_FOR_RELEASE`, clinical use, customer presentation, cloud upload, production deployment, leaderboard/SOTA equivalence.

## Current reference run

- run_id: `wmh2017_preview_20260618_e48ed25`
- release decision: `docs/release_decisions/release_decision_wmh2017_preview_20260618_e48ed25.yaml`
