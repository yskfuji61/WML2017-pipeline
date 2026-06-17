# Refactor Audit Summary — WMH2017 Dataset Layout Update

## Conclusion

Updated the Cursor/MONAI smoke scaffold to match the actual local WMH2017 Dataverse layout:

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
```

The implementation now expects:

```text
$WMH2017_ROOT/training
$WMH2017_ROOT/test
$WMH2017_ROOT/additional_annotations
```

## Key corrections

1. Dataset root is `files`, not `files/training`.
2. Manifest parser handles:
   - `training/Utrecht/<case>`
   - `training/Singapore/<case>`
   - `training/Amsterdam/<scanner>/<case>`
   - `test/Utrecht/<case>`
   - `test/Singapore/<case>`
   - `test/Amsterdam/<scanner>/<case>`
3. Additional observer annotations are detected but not used as primary smoke baseline.
4. Test cases are heldout even if `wmh.nii.gz` exists in the 2022 local release.
5. Label policy remains strict:
   - foreground: `mask == 1`
   - ignore: `mask == 2`
   - forbidden: `mask > 0`
6. Unit tests now include a synthetic WMH2017 folder structure parser test.

## Validation performed

```bash
pytest tests/unit
```

Result:

```text
14 passed
```

## Remaining non-theatrical limitations

- Real dataset scan was not executed here because the local `/Users/...` path is on the user's workstation, not in this sandbox.
- MONAI smoke training is still scaffold-level until the manifest, label audit, and split generation pass on the real local dataset.
- This is not release-ready, not clinical-use-ready, and not customer-presentation-ready.

## Recommended first local commands

```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
pytest tests/unit
```


## SOTA audit update

The latest critique correctly identifies that a MONAI 3D smoke pipeline is not enough for the shortest SOTA-candidate path. The repository now contains:
- source governance register,
- dataset card,
- metric register with all five WMH challenge metrics,
- split register,
- experiment registry,
- claim boundary register,
- failure taxonomy,
- EXP-000 and EXP-001 protocols.

Remaining hard blockers:
- sources are still marked as unresolved or requiring evidence review where exact URL/license/version must be checked by the user/team,
- official metric-code parity is not complete,
- no real dataset scan has been executed in this environment,
- no training or winner reproduction has been executed.
