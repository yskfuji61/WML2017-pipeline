# Dataset Card — MICCAI 2017 WMH Challenge

## Status

```text
REVIEW_REQUIRED
```

This card records the intended local use boundary. It is not a legal, regulatory, or clinical approval record.

## Identity

| field | value |
|---|---|
| dataset_id | WMH2017 |
| title | Data of the White Matter Hyperintensity Segmentation Challenge |
| DOI | 10.34894/AECRSD |
| publication_date | 2022-12-21 |
| license | CC-BY-NC-4.0 |
| source_register_id | SRC-WMH-DATASET-OFFICIAL |
| local_root_env_var | WMH2017_ROOT |
| expected local root | `<LOCAL_WMH2017_FILES_ROOT>` |

## Expected layout

```text
$WMH2017_ROOT/
├── training/
├── test/
├── additional_annotations/
└── readme.pdf
```

## Challenge split

| split | expected cases | use in this repo |
|---|---:|---|
| training | 60 | train/validation/smoke |
| test | 110 | heldout only; no training, validation, threshold tuning, model selection, or early stopping |
| additional_annotations | 60 x 2 observers | auxiliary analysis only; not primary baseline label |

## Modalities

Primary early pipeline:
- `pre/FLAIR.nii.gz`
- `pre/T1.nii.gz`
- `wmh.nii.gz`

Optional audit/visualization:
- `orig/FLAIR.nii.gz`
- `orig/T1.nii.gz`
- `orig/3DT1.nii.gz`

## Label policy

| label | meaning | handling |
|---:|---|---|
| 0 | background | background |
| 1 | WMH | foreground |
| 2 | other pathology | ignore |

Forbidden implementation:

```python
foreground = mask > 0
```

Required implementation:

```python
foreground = mask == 1
ignore = mask == 2
```

## DLP and use boundary

Allowed:
- local research PoC,
- internal technical validation,
- audit and reproducibility checks.

Forbidden without review:
- clinical diagnosis,
- customer-facing claim,
- commercial use,
- cloud upload,
- external API processing,
- publication of raw data or checkpoints trained on restricted data.

## Required evidence before baseline

- dataset manifest,
- strict count verification,
- label audit,
- split manifest,
- metric sanity tests,
- run manifest.
