# WMH2017 MONAI smoke training configuration

Canonical YAML: [`configs/wmh2017_monai_smoke.yaml`](../configs/wmh2017_monai_smoke.yaml)

Each run must materialize this into `artifacts/runs/{run_id}/train_config.materialized.yaml` with run-specific paths and hashes.

## YAML parameters

| Section | Key | Value | Notes |
|---|---|---:|---|
| run | seed | 42 | |
| run | device | auto | MPS uses ConvTranspose3d patch on Apple Silicon |
| data | patch_size | 32,32,32 | train crops |
| data | val_max_cases | 2 | smoke only |
| data | num_workers | 0 | |
| model | channels | 8,16,32 | |
| model | strides | 2,2 | |
| training | max_epochs | 1 | smoke |
| training | max_steps_per_epoch | 2 | smoke |
| training | learning_rate | 0.0001 | |

## Code defaults (not in YAML)

| Parameter | Value | Location |
|---|---|---|
| batch_size | 1 | `train_monai.py` |
| optimizer | Adam | `train_monai.py` |
| loss | DiceCELoss (softmax + one-hot) | `train_monai.py` |
| RandCrop pos/neg | 1 / 1 | `train_monai.py` |
| inference | sliding_window, sw_batch_size=1 | `train_monai.py` |
| normalization | `normalize_nonzero_channelwise` | preprocessing |
| label policy | foreground = label==1 only | `label_policy.py` |

## Claim boundary

This configuration validates pipeline wiring only. It is **not** a trained model configuration for performance or clinical claims.
