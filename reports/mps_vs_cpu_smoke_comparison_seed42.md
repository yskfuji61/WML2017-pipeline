# MPS vs CPU smoke comparison — seed 42

Generated: 2026-06-18 (local re-run)

## Claim boundary

This table compares **two smoke runs with different decoder implementations**, not
numerically equivalent architectures:

| run | device | decoder |
|---|---|---|
| `wmh2017_monai_smoke_seed42` | MPS (`device: auto`) | `ConvTranspose3d` → `InterpConv3d` (2 layers patched) |
| `wmh2017_monai_smoke_seed42_cpu` | CPU (`device: cpu`) | native `ConvTranspose3d` |

Same: seed 42, 2 train steps, patch 32³, val cases 107 & 110, `--allow-shape-only-geometry`.
Do **not** interpret small metric deltas as MPS correctness proof or performance claims.

## Training loss

| global_step | MPS patched (`auto→mps`) | CPU native |
|---:|---:|---:|
| 1 | 1.4370 | 1.4622 |
| 2 | 1.4191 | 1.4196 |

Artifacts:
- MPS log: `artifacts/runs/wmh2017_monai_smoke_seed42/logs/train_log.jsonl`
- CPU log: `artifacts/runs/wmh2017_monai_smoke_seed42_cpu/logs/train_log.jsonl`

## Validation Dice (local, n=2)

| metric | MPS patched | CPU native | Δ (MPS − CPU) |
|---|---:|---:|---:|
| mean_dice | **0.001753** | 0.001045 | +0.000708 |
| case 107 dice | 0.002383 | 0.001334 | +0.001049 |
| case 110 dice | 0.001123 | 0.000756 | +0.000367 |
| mean_hd95 | 125.33 | 127.58 | −2.25 |
| mean_lesion_recall | 0.974 | 0.931 | +0.043 |
| mean_lesion_f1 | 0.00124 | 0.0241 | −0.023 |

Eval split: `data/splits/wmh2017_val_with_predictions_seed42.csv` (cases 107, 110 only).

Artifacts:
- MPS metrics: `artifacts/runs/wmh2017_monai_smoke_seed42/metrics/metrics_summary.json`
- CPU metrics: `artifacts/runs/wmh2017_monai_smoke_seed42_cpu/metrics/metrics_summary.json`

## Run evidence (device / patch)

| field | MPS patched | CPU native |
|---|---|---|
| `device_selected` | mps | cpu |
| `mps_convtranspose_patched` | true | false |
| `mps_convtranspose_replaced_count` | 2 | 0 |
| `model_patch` | ConvTranspose3d_to_InterpConv3d | null |
| `native_mps_claim` | false | false |

## Interpretation

- Both runs complete end-to-end (train → predict → Dice). Pipeline wiring is validated on both paths.
- Loss at step 2 is similar (~1.419) despite different step-1 loss, reflecting the patched vs native decoder difference early in training.
- Mean Dice remains near zero on both paths, as expected for a 2-step smoke run; the MPS run is **not worse** at this smoke scale, but equivalence is **not claimed**.
- MPS path: compatibility gate passed. CPU path: release-gate smoke baseline on this machine.

## Commands used

```bash
# MPS eval (after existing MPS train)
python scripts/evaluate_wmh2017.py \
  --manifest reports/dataset_manifest.csv \
  --split data/splits/wmh2017_val_with_predictions_seed42.csv \
  --predictions artifacts/runs/wmh2017_monai_smoke_seed42/predictions \
  --out-dir artifacts/runs/wmh2017_monai_smoke_seed42/metrics \
  --run-id wmh2017_monai_smoke_seed42 \
  --allow-shape-only-geometry

# CPU train + eval
python scripts/train_wmh2017.py --config configs/wmh2017_monai_smoke_cpu_seed42.yaml
python scripts/evaluate_wmh2017.py \
  --manifest reports/dataset_manifest.csv \
  --split data/splits/wmh2017_val_with_predictions_seed42.csv \
  --predictions artifacts/runs/wmh2017_monai_smoke_seed42_cpu/predictions \
  --out-dir artifacts/runs/wmh2017_monai_smoke_seed42_cpu/metrics \
  --run-id wmh2017_monai_smoke_seed42_cpu \
  --allow-shape-only-geometry
```
