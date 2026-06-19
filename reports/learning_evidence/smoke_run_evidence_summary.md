# Training-start evidence summary

Last updated: 2026-06-19 (audit refactor; post short full run)

## Claim boundary

- Local WMH2017 PoC wiring and short/full training validation only.
- Not clinical, customer, production, SOTA, or official benchmark evidence.
- Local validation metrics are not hidden-test leaderboard results.
- Release state ceiling: **READY_FOR_PREVIEW** (see `docs/audit_gap_register.md`).

## Training-readiness wiring status (repo)

| item | status |
|---|---|
| `make e2e-full` + `EPOCHS` override | wired |
| eval `metrics_summary.json` → `run_manifest.metric_json_path` | wired |
| full-mode `model_best.pt` in e2e | wired |
| `make doctor` → `reports/env/import_smoke.json` | wired |
| full-mode first-epoch resource in `run_evidence.json` | wired |
| `@pytest.mark.requires_torch` full execution test | wired (CI) |

## Phase 0 — environment (human terminal, not Cursor sandbox)

```bash
cd "<REPO_ROOT>"
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
make doctor
python3 -c "import json; d=json.load(open('reports/env/import_smoke.json')); assert d['imports_ok']"
```

Do not start training until `imports_ok` is `true`.

## Phase 1 — tiny smoke (fresh wiring proof)

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
.venv/bin/python scripts/train_wmh2017.py --config configs/wmh2017_monai_tiny_smoke.yaml
```

**Verify locally (gitignored artifacts):**

- `artifacts/runs/wmh2017_tiny_smoke/run_evidence.json` → `status: completed`
- `artifacts/runs/wmh2017_tiny_smoke/checkpoints/model_smoke.pt` exists
- `registry/runs/run_manifest.csv` new row with `status=completed` (not `artifacts_local_expired`)

| metric (fill after run) | value |
|---|---:|
| run_id | `wmh2017_tiny_smoke` |
| global_step | _TBD_ |
| val_prediction_count | _TBD_ |
| device | _TBD_ |

## Phase 2 — short full + eval (3 epochs)

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
make e2e-full EPOCHS=3 RUN_ID=wmh2017_full_short_seed42 WMH2017_ROOT="$WMH2017_ROOT" ALLOW_DIRTY_GIT=1 OVERWRITE_RUN=1
```

**Verify locally:**

- `artifacts/runs/wmh2017_full_short_seed42/run_evidence.json`
  - `resource.first_epoch.wall_time_seconds`
  - `resource.first_epoch.peak_rss_kb`
- `artifacts/runs/wmh2017_full_short_seed42/evaluation/metrics_summary.json`
- `registry/runs/run_manifest.csv` row for `wmh2017_full_short_seed42`:
  - `status=completed`
  - `metric_json_path` points to evaluation metrics summary (relative/redacted in commit)
  - `checkpoint_hash` non-empty

| metric (fill after run) | value |
|---|---:|
| run_id | `wmh2017_full_short_seed42` |
| max_epochs (actual) | 3 |
| run_purpose (expected after audit refactor) | `wmh2017_monai_full_training` |
| best_val_dice | 0.0171 (epoch 1) |
| mean_dice (eval) | 0.0171 |
| val_dice by epoch | 0.01225 → 0.01710 → 0.01449 (noise, not convergence) |
| first_epoch_wall_s | 114.2 |
| first_epoch_peak_rss_kb | 2575296 (~2.46 GB) |
| device | mps (float32; no MPS fp16 autocast) |
| global_step | 144 |

### Interpretation limits (required; not performance claims)

- **Dice ≈ 0.017** confirms end-to-end wiring on real WMH2017 data; it is **not** segmentation quality evidence.
- Three epochs are insufficient for convergence; val Dice fluctuates rather than improving monotonically.
- **`lesion_recall` ≈ 0.93** with **`pred_lesion_count` ≫ `target_lesion_count`** indicates over-segmentation; recall alone is misleading.
- **`hd95` / `avd`** were computed under **`geometry_policy: shape_only`** (`geometry_metrics_physical_units: false`); not valid in physical mm/mm³.
- **MPS path** substitutes decoder **`ConvTranspose3d` → `InterpConv3d`** (`architecture_parity.comparable_to_native_convtranspose: false`); not numerically comparable to native ConvTranspose on CUDA/CPU.
- **`claim_allowed.leaderboard_or_sota: false`** in local `metrics_summary.json`; do not cite these numbers externally.

### Resource extrapolation (after Phase 2)

Use measured epoch-0 values from `run_evidence.json`:

```
estimated_full_50ep_hours ≈ (first_epoch_wall_s / 3600) × 50  ≈ 1.59 h (train epoch 0 only; val sliding-window not included)
estimated_peak_rss_gb ≈ first_epoch_peak_rss_kb / (1024 × 1024)  ≈ 2.46 GB
```

Record extrapolation here after a successful short full run; do not commit absolute local paths.

## Prior smoke reference (expired on disk)

Historical EXP-000 wiring check (artifacts no longer on disk; `run_manifest` rows marked `artifacts_local_expired`):

| metric | value |
|---|---:|
| n_cases (val with predictions) | 2 |
| mean_dice | 0.00105 |
| mean_hd95 | 127.58 |
| global_step | 2 |

Near-zero Dice is expected for a 2-step smoke run; confirms wiring only.

## Artifacts policy

- Checkpoints, predictions (`.nii.gz`), and raw `case_metrics.csv` remain **gitignored** under `artifacts/runs/`.
- Hashes and status are recorded in `registry/runs/run_manifest.csv` after local runs.
- `reports/env/import_smoke.json` is gitignored; verify locally only.
- Commit only redacted metrics and relative artifact paths in this file and `run_manifest.csv`.

## Post-run commit checklist (Cursor)

1. Update metric tables above with redacted values only.
2. Update `registry/runs/run_manifest.csv` if paths need redaction (no absolute home paths).
3. `git status` — ensure no `.pt`, `.nii.gz`, `.secrets.baseline`, or `reports/release_package_manifest.json`.
4. Commit: `docs: update learning evidence with fresh training runs`
