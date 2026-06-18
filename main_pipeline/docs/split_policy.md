# Split Policy — WMH2017

## Conclusion

Create train/validation only from challenge `training` cases. Treat challenge `test` as heldout evaluation only, even when the local 2022 release includes test labels.

## Root and split source

Recommended dataset root:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

Expected source split directories:

```text
$WMH2017_ROOT/training
$WMH2017_ROOT/test
$WMH2017_ROOT/additional_annotations
```

## Allowed initial split

- Eligible for train/validation: `challenge_split == training` and primary `wmh.nii.gz` exists
- Not eligible for train/validation: `challenge_split == test`
- Not eligible as primary reference in smoke baseline: `additional_annotations/**/result.nii.gz`

## Test split rule

Even if `files/test/**/wmh.nii.gz` exists in the local 2022 release, test cases must not be used for:

- training
- validation
- threshold tuning
- preprocessing fit
- model selection
- early stopping

Any metric computed on test cases must be labelled as `local heldout evaluation` and must not be mixed with:

- challenge leaderboard results
- training validation results
- clinical or customer-facing claims

## Stratification

Use `site` and `scanner_code` as stratification candidates when possible. If the first smoke run is too small for stratification, record that limitation in `split_summary.json`.

## split_manifest columns

```csv
split_id,case_id,challenge_split,source_split,assigned_split,site,scanner,scanner_code,group_id,seed,reason,created_at
```

## Leakage conditions

The following are leakage or invalid conditions:

- test case assigned to train or val
- test case used for threshold tuning
- test case used for preprocessing fit
- additional observer annotation used as primary reference without explicit experiment ID
- same source case appears in both train and val
- validation result is reported as final test performance
- local heldout result is reported as official challenge result

## Invalid claim conditions

- no run_id
- no split_id
- no metric_id
- no config hash
- no dataset manifest hash
- no environment record
- no label policy record
- test metric described without metric population
