# Local environment setup (WMH2017 PoC)

This guide covers reproducible local setup for public WMH2017 smoke and training runs.
Raw challenge data must never be committed to git.

## 1. Python environment

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e ".[dev,test,medical-image]"
make doctor
```

If `make doctor` reports import segfault/SIGABRT, recreate `.venv` and ensure you are
using a supported Python (3.10–3.11). Do not mix conda and `.venv` interpreters.

The doctor also writes `reports/env/import_smoke.json` (gitignored) with import results
and MPS availability. Confirm `"imports_ok": true` before starting training runs.

`imports_ok` covers the Python import stack only. `WMH2017_ROOT` is checked separately:
do **not** export doc placeholders such as `<LOCAL_WMH2017_FILES_ROOT>` literally — either unset
it for `make doctor`, or point it at your local `files/` directory before `make e2e`.

## 2. WMH2017_ROOT

Point `WMH2017_ROOT` at the Dataverse **`files`** directory (not `training/`):

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Example layout under `$WMH2017_ROOT`:

- `training/<site>/<case_id>/pre/FLAIR.nii.gz`
- `training/<site>/<case_id>/wmh.nii.gz`
- `test/...` (held out; never used for train/val/tuning)

## 3. Canonical manifests

| Artifact | Role |
|---|---|
| `reports/dataset_manifest.csv` | **Canonical** local dataset index (gitignored; regenerate locally) |
| `data/splits/wmh2017_train_val_seed42.csv` | **Canonical** train/val split (tracked) |
| `artifacts/manifests/*.json` | Redacted v4 derivatives (sync from CSV/split) |

Regenerate v4 JSON manifests after updating local CSV:

```bash
.venv/bin/python scripts/data/sync_v4_manifests_from_csv.py \
  --dataset-csv reports/dataset_manifest.csv \
  --split-csv data/splits/wmh2017_train_val_seed42.csv
```

## 4. Smoke run (wiring check)

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
make e2e RUN_ID=wmh2017_smoke_local WMH2017_ROOT="$WMH2017_ROOT"
```

Or use the tiny wrapper:

```bash
./scripts/run_wmh2017_tiny_smoke.sh \
  --root "$WMH2017_ROOT" \
  --run-id wmh2017_tiny_smoke \
  --config configs/wmh2017_monai_tiny_smoke.yaml
```

## 5. Full training run (train + eval chain)

Use `make e2e-full` for the full MONAI 3D U-Net config with evaluation and manifest wiring.
The default config is `configs/wmh2017_monai_unet3d_full.yaml` (50 epochs). Override epochs for
short validation runs without editing the config:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
make e2e-full RUN_ID=wmh2017_full_short_seed42 EPOCHS=3 WMH2017_ROOT="$WMH2017_ROOT"
```

Dry-run the command expansion without executing training:

```bash
make -n e2e-full EPOCHS=3 RUN_ID=wmh2017_full_short_seed42 WMH2017_ROOT="$WMH2017_ROOT"
```

## 6. Platform notes (torch / MPS / CUDA)

- `requirements-lock.txt` pins versions but not wheel hashes; torch wheels differ by platform.
- Record the approved wheel index in run evidence when using CUDA.
- On Apple Silicon, MONAI smoke uses an MPS ConvTranspose3d compatibility patch; this is
  wiring validation only, not native-MPS equivalence.

## 7. Prohibited without review

- Proprietary/private/PHI data
- Cloud upload of medical data
- Customer-facing or clinical readiness claims
- Official benchmark / leaderboard equivalence claims
