# Architecture — Lightweight MLOps Evidence MONAI Smoke Pipeline

## Conclusion

This repository should implement a local research PoC pipeline, not a clinical product and not a production MLOps platform.

Initial objective:

1. Generate a WMH2017 case manifest from the Dataverse `files` root.
2. Confirm label policy and split policy.
3. Run MONAI smoke training on challenge training cases only.
4. Record enough run evidence to make later MLOps audit possible.

## Dataset root

Use the Dataverse `files` directory as root:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Do not point root to `training` only. The scanner needs access to `training`, `test`, and `additional_annotations` to prevent leakage and record evidence.

## Main modules

```text
src/wmh2017/data/manifest.py       # Dataverse layout parser and case manifest
src/wmh2017/data/label_policy.py   # label==1 foreground, label==2 ignore
src/wmh2017/data/splits.py         # training-only train/val split
src/wmh2017/evaluation/            # metrics and golden tests
src/wmh2017/audit/                 # run evidence helpers
scripts/                           # CLI entry points
registry/                          # CSV schemas and evidence records
docs/                              # policies and Cursor handoff
```

## MONAI usage boundary

Use MONAI for:

- Dataset/CacheDataset/DataLoader integration
- transforms
- 3D U-Net smoke model
- sliding window inference later
- Dice loss / Dice metric where appropriate

Do not use MONAI abstractions to hide dataset split policy, label policy, or evidence recording.

## Primary input policy

Initial smoke training should use:

```text
pre/FLAIR.nii.gz
pre/T1.nii.gz
training/**/wmh.nii.gz
```

`orig/` can be used later for experiments but should not be the first baseline unless explicitly recorded.

## Label policy

- `mask == 1`: WMH foreground
- `mask == 2`: other pathology / ignore
- `mask > 0`: forbidden

## Split policy

- `training`: train/val eligible
- `test`: heldout only
- `additional_annotations`: auxiliary only in initial phase

## Future MLOps extension points

After baseline:

- model registry
- experiment comparison
- model card
- approval workflow
- rollback plan
- monitoring plan
- drift and performance dashboard

Do not build these before smoke and baseline evidence exists.
