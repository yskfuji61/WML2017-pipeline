# Baseline comparison report

## Conclusion

Smoke-run metrics document local MONAI U-Net baseline behavior on the preview run. This report does **not** make performance claims against challenge leaders or clinical benchmarks.

## Preview run

| Field | Value |
|-------|-------|
| run_id | `wmh2017_preview_20260618_e48ed25` |
| architecture | MONAI 3D U-Net smoke (see model card) |
| scope | Local validation + heldout eval labels where permitted |

Metrics are recorded in `artifacts/runs/<run_id>/evaluation/` (gitignored; hashes in release evidence register).

## Blocked claims

SOTA, leaderboard rank, clinical efficacy, and generalization to unseen populations remain blocked.
