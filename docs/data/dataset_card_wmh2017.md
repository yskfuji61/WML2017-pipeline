# WMH2017 public dataset card (v4)

## Scope

Public WMH2017 challenge data only. Local root via `WMH2017_ROOT`. No raw MRI committed.

## Manifest

Generate with:

```bash
python scripts/data/generate_dataset_manifest.py --root "$WMH2017_ROOT" --output artifacts/manifests/dataset_manifest.json
```

Paths in committed JSON are redacted (`REDACTED_OR_LOCAL_ONLY`).

## Label policy

`LABEL_POLICY_PENDING` until label audit completes.

## DLP

No proprietary/private/PHI/PII data use, storage, model training, export, upload, or report inclusion.
