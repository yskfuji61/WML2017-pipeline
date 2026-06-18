# Cursor Task Plan — WMH2017 MONAI Smoke Pipeline

## Phase 0: repo clean / scaffold

objective: Confirm this repo contains no raw NIfTI/checkpoint/prediction data.

files to create/edit:
- .gitignore
- README_CURSOR_START.md

commands:
```bash
git status
find . -name '*.nii' -o -name '*.nii.gz' -o -name '*.dcm'
```

acceptance criteria:
- raw dataset is outside repo
- no raw image files are tracked

stop conditions:
- raw medical image files are inside repo

rollback:
- remove generated/raw files from repo, keep raw data under `$WMH2017_ROOT`

## Phase 1: source and dataset manifest

objective: Scan the Dataverse `files` layout and create one row per case.

commands:
```bash
export WMH2017_ROOT="<LOCAL_WMH2017_FILES_ROOT>"
python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
```

acceptance criteria:
- 60 training cases
- 110 test cases
- Utrecht/Singapore/Amsterdam site distribution recorded
- Amsterdam scanner subfolders parsed
- O3/O4 availability recorded for training cases

stop conditions:
- root points to `training` instead of `files`
- strict counts fail

## Phase 2: label audit and overlay

objective: Confirm primary training masks follow `{0,1,2}` and label 2 is not foreground.

commands:
```bash
python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
python scripts/visualize_wmh_case.py --manifest reports/dataset_manifest.csv --case-id 0 --out reports/overlays
```

acceptance criteria:
- no labels outside `{0,1,2}`
- overlay uses `mask == 1`
- label 2 count is recorded, not treated as WMH

## Phase 3: split generation

objective: Create train/val from challenge training only and mark challenge test as heldout.

commands:
```bash
python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
```

acceptance criteria:
- no `challenge_split=test` in train/val
- `data/splits/wmh2017_test110_heldout.csv` contains heldout rows only

## Phase 4: MONAI Dataset/DataLoader

objective: Implement MONAI dataset using `pre/FLAIR.nii.gz`, `pre/T1.nii.gz`, and primary `wmh.nii.gz`.

files:
- src/wmh2017/data/monai_dataset.py
- tests/unit/test_monai_dataset_smoke.py

acceptance criteria:
- loads at least one training case
- channels are explicit
- mask foreground policy is applied later, not hidden by `mask > 0`

## Phase 5: metric golden tests

objective: Lock Dice behavior and prepare HD95/AVD/lesion metrics.

commands:
```bash
pytest tests/unit/test_metrics_golden.py
```

acceptance criteria:
- perfect match Dice = 1
- empty prediction with nonempty target near 0
- label 2 is not foreground

## Phase 6: smoke training

objective: Run a small MONAI 3D U-Net smoke training.

acceptance criteria:
- forward/backward pass works
- loss is finite
- validation Dice is produced
- no performance claim beyond smoke

## Phase 7: validation evaluation

objective: Evaluate validation only, save metric JSON and overlay samples.

acceptance criteria:
- metric report states split and case count
- no test result mixed with validation

## Phase 8: run manifest and report

objective: Record run evidence.

acceptance criteria:
- run_id
- config_hash
- dataset_manifest_hash
- split_manifest_hash
- MONAI/PyTorch versions
- metric JSON path

## Phase 9: baseline extension

objective: Move from smoke to baseline only after Phases 1-8 pass.

forbidden before this phase:
- nnU-Net comparison
- ensemble
- customer-facing report
- SOTA claim


---

# SOTA Candidate Audit Extension

## Stage 0: source / claim freeze

objective:
- Replace placeholder sources before SOTA-oriented comparison.
- Freeze metric IDs, split IDs, and claim boundary.

files:
- `registry/source_register_wmh2017.csv`
- `registry/dataset_card_wmh2017.md`
- `registry/metric_register_wmh2017.csv`
- `registry/split_register_wmh2017.csv`
- `registry/claim_boundary_wmh2017.csv`

acceptance criteria:
- no SOTA claim wording appears in README/reports
- source statuses are reviewed before use in comparison
- metric IDs are fixed
- split IDs are fixed

stop conditions:
- source license unclear
- official metric code not found but benchmark claim is requested

## EXP-000: data integrity + metric sanity + visualization

objective:
- Generate the first valid audit run before training.

commands:
```bash
python scripts/audit_wmh2017_dataset.py --root "$WMH2017_ROOT" --out reports/dataset_manifest.csv --strict-counts
python scripts/audit_wmh2017_labels.py --manifest reports/dataset_manifest.csv --split training --out reports/label_value_audit.csv
python scripts/make_wmh2017_splits.py --manifest reports/dataset_manifest.csv --seed 42 --out-dir data/splits
pytest tests/unit
```

acceptance criteria:
- training=60 and test=110
- label values within `{0,1,2}`
- label 2 ignored
- test rows heldout only
- DSC/H95/AVD/lesion recall/lesion F1 golden tests pass

## EXP-001: winner reproduction planning

objective:
- Verify source, license, model provenance, metric compatibility, and preprocessing deviations before running winner reproduction.

files:
- `docs/future_sota/EXP-001_winner_reproduction_plan.md`
- `registry/source_register_wmh2017.csv`
- `registry/experiment_registry_wmh2017.csv`

acceptance criteria:
- source IDs verified
- implementation license reviewed
- exact input/preprocessing/threshold/postprocess policy documented
- no claim stronger than `winner reproduction attempt`

stop conditions:
- source/license ambiguity
- pretrained model provenance unclear
- official metric parity not documented
