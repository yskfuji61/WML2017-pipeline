# Release state crosswalk

## Purpose

This file removes ambiguity between the project-local planning term
`READY_FOR_REQUIREMENTS_REVIEW` and the controlled audit release states used for
external evidence review.

## Controlled status for this package

```text
package_status: STRUCTURAL_REVIEW_IN_PROGRESS
controlled_release_state: STRUCTURAL_REVIEW
release_ladder: STRUCTURAL_REVIEW -> PREVIEW_CANDIDATE -> READY_FOR_PREVIEW -> LIMITED_INTERNAL_USE -> READY_FOR_RELEASE
upper_bound_without_real_run_evidence: STRUCTURAL_REVIEW
upper_bound_without_preview_or_human_review: READY_FOR_PREVIEW
ready_for_release: false
historical_preview_run_id: wmh2017_preview_20260618_e48ed25
v4_intermediate_states: READY_FOR_STRUCTURAL_REVIEW, REAL_SMOKE_RUN_EVIDENCE_AVAILABLE, STRUCTURAL_REVIEW_HARDENED, REAL_WMH2017_SMOKE_RUN_EVIDENCE_AVAILABLE
```

## v4 intermediate states (additive)

| v4 state | Meaning | Maps to controlled ladder |
|---|---|---|
| READY_FOR_STRUCTURAL_REVIEW | Alias of STRUCTURAL_REVIEW | STRUCTURAL_REVIEW |
| REAL_SMOKE_RUN_EVIDENCE_AVAILABLE | Tiny real local smoke succeeded | PREVIEW_CANDIDATE or higher |
| STRUCTURAL_REVIEW_HARDENED | CI/security gates hardened | PREVIEW_CANDIDATE |
| REAL_WMH2017_SMOKE_RUN_EVIDENCE_AVAILABLE | Full minimal WMH2017 smoke evidence | READY_FOR_PREVIEW candidate |

Historical preview evidence (`wmh2017_preview_20260618_e48ed25`) is retained during v4 alignment.

## State mapping

| Local phrase | Controlled meaning | Allowed use | Forbidden interpretation |
|---|---|---|---|
| READY_FOR_REQUIREMENTS_REVIEW | Internal pre-structural state. Scope, stop conditions, and evidence plan are present; evidence is not complete. | June catch-up planning and local implementation routing. | Preview-ready, customer-ready, validated, approved, clinical, or production-ready. |
| READY_FOR_STRUCTURAL_REVIEW | Same as STRUCTURAL_REVIEW (legacy alias). Static structure can be checked after full package manifest, file inventory, source register, and test plan exist. | Repository quality review only. | Any claim that WMH2017 training/evaluation has succeeded. |
| PREVIEW_CANDIDATE | Real run evidence + fail-closed security; reviewers assigned but approval may be pending. | Internal preview preparation only. | Preview-approved or release claim. |
| READY_FOR_PREVIEW | Real public-data evidence exists and can be reviewed in a limited internal preview. | Limited internal technical preview only. | Customer-facing, clinical, proprietary-data, cloud, or SOTA claim. |
| READY_FOR_LIMITED_INTERNAL_USE | Preview findings are reviewed, no Sev0/Sev1 remains, and residual Sev2 is accepted by named owners. | Bounded internal technical use. | Release, diagnostic, customer or clinical performance claim. |
| READY_FOR_RELEASE | Requires exact frozen package identity, complete manifest/hash, source review, real run evidence, review disposition, approvals, release decision, rollback plan, and applicable human signoffs. | Only after formal release decision. | Must never be inferred from static files or smoke tests alone. |

## Current blocked claims

- clinical use
- diagnostic use
- reader replacement
- patient-level risk judgment
- customer presentation
- production deployment
- proprietary/private data processing
- unapproved cloud upload
- SOTA, leaderboard, or official benchmark equivalence

## Promotion gate

Promotion is blocked until all open Sev1 findings in `registry/finding_register_wmh2017.csv` are closed with evidence, owner, date, and reviewer disposition.
