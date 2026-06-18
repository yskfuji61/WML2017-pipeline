# Source and license review checklist

## Purpose

Checklist for human review of WMH2017 public data, official evaluation material,
papers, MONAI/PyTorch dependencies, and any reused implementation. This document
does not complete the review by itself.

## Required source review fields

For each source in `registry/source_register_wmh2017.csv`, complete:

- source_id
- exact title
- owner/organization
- source type
- authority level
- URL or DOI
- version or publication date
- access date / last verified
- review due
- license or terms
- allowed uses
- forbidden uses
- commercial/customer boundary
- redistribution boundary
- citation requirement
- affected files
- affected claims
- reviewer
- reviewer qualification
- review date
- disposition
- replacement/escalation path

## Immediate blockers

- Any dataset or code source used before license/terms review.
- Any leaderboard/SOTA comparison before official evaluator and protocol review.
- Any customer-facing statement based on source reports without reviewer-approved claim boundary.
- Any reused implementation without license/provenance/version review.
- Any claim that a paper result is reproduced without matching data, split, metric, implementation, and run evidence.
