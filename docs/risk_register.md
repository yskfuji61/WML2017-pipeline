# Risk Register — WMH2017 MONAI Smoke Pipeline

| risk_id | severity | risk | trigger | mitigation | test/evidence | owner |
|---|---:|---|---|---|---|---|
| R-001 | Sev1 | challenge test cases enter train/val/tuning | `challenge_split=test` assigned to train/val | `make_train_val_split` hard guard | `test_split_no_leakage.py` | implementation_lead |
| R-002 | Sev1 | label 2 treated as WMH foreground | implementation uses `mask > 0` | `wmh_foreground_mask(mask == 1)` only | `test_label_policy.py` | implementation_lead |
| R-003 | Sev1 | local test labels reported as validation or official score | evaluating `files/test/**/wmh.nii.gz` without population label | require `local heldout evaluation` wording | metric report review | evidence_reviewer |
| R-004 | Sev2 | Dice-only overclaim | reporting Dice as sufficient performance | add HD95/AVD/lesion recall/F1 before comparison | `metric_policy.md` | evidence_reviewer |
| R-005 | Sev2 | raw NIfTI enters git | `git add Datasets/` or raw paths under repo | `.gitignore`, raw root outside repo | git status review | implementation_lead |
| R-006 | Sev2 | cloud upload without approval | local GPU insufficient | stop condition and approval gate | review note | security_privacy_reviewer |
| R-007 | Sev2 | additional observer annotations used as primary baseline | `additional_annotations/**/result.nii.gz` used in training | manifest records O3/O4 but smoke excludes | manifest/schema review | medical_domain_reviewer |
| R-008 | Sev2 | preprocessing leakage | fit normalization/resampling decisions on test | restrict fit to training split | config/review | implementation_lead |
| R-009 | Sev2 | scanner/site imbalance hidden | random split without site/scanner summary | record site/scanner in manifest and split | split_summary.json | evidence_reviewer |
| R-010 | Sev3 | workstation absolute path committed | hardcoded `/Users/...` in code | keep path only in local env/docs example | grep review | implementation_lead |
| R-011 | Sev3 | downloaded evidence mismatches dataset | incomplete download | strict count and checksum/size evidence | download_record, strict-counts | implementation_lead |
| R-012 | Sev1 | clinical/customer misuse | PoC results shown as diagnosis or customer performance | prohibited-use copy and review gate | release review | owner |
