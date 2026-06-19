# Metric protocol (v4)

Local validation metrics are computed from prediction masks and WMH label==1 foreground policy.
Official WMH2017 hidden-test benchmark results must not be inferred from local validation.

## Required local metrics

- dice_local
- hd95_local_mm
- avd_local_percent
- lavd_wmh2017_compat_candidate
- lesion_recall_local
- lesion_f1_local

## Separation rule

Local AVD is not official lAVD. Leaderboard or SOTA claims remain blocked without reviewed parity evidence.
