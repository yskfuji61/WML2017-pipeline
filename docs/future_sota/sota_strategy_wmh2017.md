# WMH2017 SOTA Candidate Strategy and Audit Boundary

## Conclusion

This repository is prepared for building **SOTA candidates**, not for claiming SOTA.

Current release state remains:

```text
READY_FOR_REQUIREMENTS_REVIEW
```

The next valid run is:

```text
EXP-000: data integrity + metric sanity + 1-case visualization
```

No training, winner reproduction, MONAI/nnU-Net comparison, or SOTA comparison should start until EXP-000 passes.

## Fixed scope

| item | fixed boundary |
|---|---|
| task | White Matter Hyperintensity segmentation |
| modality | FLAIR primary, T1 optional/standard comparison input |
| data | MICCAI 2017 WMH Challenge public Dataverse release |
| non-goals | diagnosis, clinical decision support, customer presentation, production deployment |
| primary metric | MET-DSC |
| required secondary metrics | MET-H95, MET-AVD, MET-LESION-RECALL, MET-LESION-F1 |
| release state | READY_FOR_REQUIREMENTS_REVIEW |

## Why the previous MONAI-only plan is not enough for SOTA proximity

MONAI 3D U-Net is a valid engineering baseline and satisfies the meeting requirement to run a 3D segmentation pipeline. It is not, by itself, the shortest route to a SOTA candidate.

The SOTA-oriented path must compare:
1. winner-style 2D FCN ensemble reproduction,
2. MONAI 2D baseline,
3. MONAI 2.5D baseline,
4. nnU-Net v2 2D/3D baselines,
5. MONAI 3D U-Net / SegResNet candidates.

The first objective is not to beat the leaderboard. It is to create a valid reproduction baseline with source, split, metric, config, environment, and artifact evidence.

## Required stages

### Stage 0: project freeze

Required outputs:
- `registry/source_register_wmh2017.csv`
- `registry/dataset_card_wmh2017.md`
- `registry/metric_register_wmh2017.csv`
- `registry/split_register_wmh2017.csv`
- `registry/experiment_registry_wmh2017.csv`
- `registry/claim_boundary_wmh2017.csv`
- `registry/failure_taxonomy_wmh2017.csv`

Pass conditions:
- no unreviewed source is used for claims,
- primary/secondary metrics are fixed,
- split IDs are fixed,
- claim boundary is explicit,
- DLP class is recorded.

### Stage 1: EXP-000 data integrity

Required outputs:
- dataset manifest,
- checksum/evidence summary if hashing is enabled,
- modality pairing table,
- scanner/site count table,
- mask value report,
- one FLAIR/T1/mask overlay,
- metric sanity test result,
- run evidence record.

Pass conditions:
- training=60, test=110 under strict count mode,
- labels are within `{0, 1, 2}` for training masks,
- label 2 is ignored, not foreground,
- metrics pass golden tests,
- run_id and hashes are recorded.

### Stage 2: EXP-001 winner reproduction

Do this only after source/license review.

Required outputs:
- source verification record,
- winner preprocessing config,
- pretrained inference or reimplementation note,
- local validation metrics,
- scanner-wise metric table,
- deviation log.

Do not call this SOTA. Use `reproduction candidate` or `local reproduction attempt`.

### Stage 3: MONAI/nnU-Net comparison

Compare only under:
- same dataset version,
- same split,
- same metrics,
- same threshold policy,
- same postprocess boundary,
- scanner-wise reporting.

### Stage 4: SOTA candidate optimization

Allowed exploration:
- FLAIR vs FLAIR+T1,
- 2D / 2.5D / 3D,
- Dice / DiceCE / Focal Tversky,
- lesion oversampling,
- augmentation,
- validation-only thresholding,
- pre-registered ensemble/TTA,
- pre-registered postprocess.

Forbidden:
- using test for tuning,
- changing metrics after seeing results,
- cherry-picking scanner subsets,
- comparing local validation to official hidden-test leaderboard as if equal.

## SOTA claim boundary

SOTA claim is blocked unless all are true:

1. `source_register_wmh2017.csv` has verified source IDs.
2. Dataset version and license/use boundary are reviewed.
3. Split ID is fixed before training.
4. Metric implementation is either official or parity-checked.
5. Run evidence includes config/code/env/dataset/split hashes.
6. Findings Sev0/Sev1 are closed.
7. Evidence/domain/security reviewers approve claim wording.
8. Comparison validity record exists.

Until then, allowed wording is:

```text
SOTA candidate plan
local validation result
source-reported leaderboard value
winner reproduction attempt
```

Forbidden wording:

```text
SOTA achieved
clinical-grade
production-ready
diagnostic performance
official benchmark result
```
