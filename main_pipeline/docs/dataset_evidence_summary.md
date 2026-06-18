# Dataset Evidence Summary — WMH2017

## Conclusion

The local dataset root should be the Dataverse `files` directory, not the `training` directory.

Recommended local environment variable:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Expected top-level structure:

```text
$WMH2017_ROOT/
├── training/
├── test/
├── additional_annotations/
└── readme.pdf
```

## Confirmed challenge structure from readme

- Training: 60 cases
  - UMC Utrecht / 3T Philips Achieva: 20
  - NUHS Singapore / 3T Siemens TrioTim: 20
  - VU Amsterdam / 3T GE Signa HDxt: 20
- Test: 110 cases
  - UMC Utrecht / 3T Philips Achieva: 30
  - NUHS Singapore / 3T Siemens TrioTim: 30
  - VU Amsterdam / 3T GE Signa HDxt: 30
  - VU Amsterdam / 3T Philips Ingenuity: 10
  - VU Amsterdam / 1.5T GE Signa HDxt: 10

## Per-case files

Primary smoke-training inputs:

```text
pre/FLAIR.nii.gz
pre/T1.nii.gz
wmh.nii.gz
```

Useful optional files:

```text
orig/FLAIR.nii.gz
orig/T1.nii.gz
orig/3DT1.nii.gz
orig/reg_3DT1_to_FLAIR.txt
```

## Label policy

Manual reference standard labels:

| label | meaning | smoke pipeline handling |
|---:|---|---|
| 0 | Background | background |
| 1 | White matter hyperintensities | foreground |
| 2 | Other pathology | ignore |

Forbidden implementation:

```python
foreground = mask > 0
```

Required implementation:

```python
foreground = mask == 1
ignore = mask == 2
```

## Important 2022 release caveat

The original challenge readme describes `wmh.nii.gz` as available only for training data, because the challenge test data was originally hidden.

If the 2022 Dataverse release contains `files/test/**/wmh.nii.gz`, those labels still must not be used for:

- training
- validation
- threshold tuning
- preprocessing fit
- model selection
- early stopping

Any metric computed on `test` must be labelled as `local heldout evaluation`, not as `training validation`, not as `official challenge score`, and not as a customer-facing claim.

## Additional annotations

Additional observer annotations are auxiliary evidence:

```text
additional_annotations/observer_o3/training/**/result.nii.gz
additional_annotations/observer_o4/training/**/result.nii.gz
```

Initial MONAI smoke/baseline must use the primary `training/**/wmh.nii.gz` reference only.

Use O3/O4 later for:

- inter-observer variability
- annotation uncertainty
- robustness analysis


## Captured download evidence

The package now contains non-raw official-download evidence at:

```text
evidence/wmh2017_download_2026-06-16/
```

Use this command to verify the packaged metadata/checksum evidence:

```bash
python scripts/verify_wmh2017_download_evidence.py   --evidence-dir evidence/wmh2017_download_2026-06-16   --out reports/wmh2017_download_evidence_verification.json
```

This is acquisition-integrity evidence only. The actual raw dataset remains
external to the repository and must be referenced by a local `WMH2017_ROOT`.
