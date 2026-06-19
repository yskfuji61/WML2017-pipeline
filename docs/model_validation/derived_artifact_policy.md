# Derived artifact policy (v4)

## Git tracking rules

| artifact | git policy |
|---|---|
| Raw WMH2017 volumes (`.nii.gz`, `.zip`) | **never commit** |
| Checkpoints (`.pt`) | **never commit** |
| Prediction masks under `artifacts/runs/` | **never commit** |
| Raw `case_metrics.csv` / full evaluation dirs | **never commit** (gitignored) |
| Redacted summaries (`reports/learning_evidence/*.md`) | may commit after path/overclaim review |
| v4 JSON manifests (`artifacts/manifests/*.json`) | commit redacted derivatives only |

## Redaction requirements

- Overlay PNG and error map PNG are medical-image-derived artifacts.
- Generated PNGs under `reports/overlays/` must not be committed by default.
- Absolute local paths must not appear in tracked docs, registers, or JSON manifests.
- Use `scripts/data/sync_v4_manifests_from_csv.py` to regenerate redacted JSON from canonical CSV.
- Run `scripts/verify_no_absolute_path_leakage.py` before committing evidence docs.

## Claims

- Public WMH2017 derived figures and metrics summaries are local PoC evidence only.
- They do not imply clinical quality, customer readiness, or official benchmark equivalence.

