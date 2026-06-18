# EXP-001 Plan — Winner FCN Ensemble Reproduction

## Objective

Create a reproducible local reproduction attempt of the MICCAI 2017 winner-style FCN ensemble after EXP-000 passes.

## Claim boundary

This experiment may produce:
- `winner reproduction attempt`,
- `local validation result`,
- `source-reported comparison`.

It must not produce:
- `SOTA achieved`,
- `official benchmark result`,
- `clinical performance`,
- `production-ready`.

## Required before start

- `SRC-LI-FCN-ENSEMBLE-2018` verified,
- implementation/code/model license reviewed,
- official metric code located or metric parity plan documented,
- split ID fixed,
- preprocessing config frozen,
- threshold/postprocess policy frozen.

## Candidate inputs

- FLAIR only,
- FLAIR + T1.

## Required outputs

- source verification note,
- implementation provenance note,
- config hash,
- environment hash,
- validation metrics for all five metrics,
- scanner-wise metric table,
- deviation report against source-reported method.

## Stop conditions

- license is unclear,
- pretrained model provenance is unclear,
- metric implementation cannot be matched or documented,
- result is being compared to official leaderboard without boundary text.
