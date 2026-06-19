# Current repository state

Last updated: 2026-06-19 (audit refactor FIX-1)

## Manifest canonical sources

| Artifact | Role |
|---|---|
| `reports/dataset_manifest.csv` | **Canonical** local dataset index (gitignored; regenerate locally) |
| `data/splits/wmh2017_train_val_seed42.csv` | **Canonical** train/val split (tracked) |
| `artifacts/manifests/*.json` | Redacted v4 derivatives via `scripts/data/sync_v4_manifests_from_csv.py` |

## Controlled release state

```text
controlled_release_state: READY_FOR_PREVIEW
historical_preview_run_id: wmh2017_preview_20260618_e48ed25
v4_progress_state: READY_FOR_STRUCTURAL_REVIEW
v4_phases_completed: PR-0,PR-1,PR-SMOKE-1,PR-2,PR-3,PR-4,PR-5,PR-6,PR-7,WAVE-3
ready_for_release: false
```

Historical preview evidence from Wave 1–2 is **retained**. v4 intermediate states are tracked in
[`release_state_crosswalk.md`](../release_state_crosswalk.md).

## Allowed manager wording (current)

- Public WMH2017 local PoC technical verification is in progress.
- Preview structural package exists with limited internal review evidence.
- Local validation metrics are not official hidden-test benchmark results.

## Prohibited claims

- READY_FOR_RELEASE
- clinical / customer / production readiness
- SOTA or official benchmark equivalence
- AI diagnosis capability

## DLP rule (v4)

No proprietary/private/PHI/PII data use, storage, model training, export, upload, or report inclusion.
Local redacted metadata inspection for DLP/security review is allowed only to determine whether review is required.
Raw metadata values must not be printed, committed, logged, exported, or included in reports.
If PHI/PII-like metadata is found, stop and require security/privacy review.
