# TEST_PLAN

## test_label_policy.py

- purpose: label==1 foreground、label==2 ignoreを固定する
- fixture: numpy arrays with labels 0/1/2 and invalid label 3
- expected result: label==2 foreground count is 0
- failure condition: `mask > 0`と同じ挙動になる

## test_split_no_leakage.py

- purpose: test source splitがtrain/valに混入しないことを確認する
- fixture: dummy manifest with training/test rows
- expected result: test rows assigned to heldout_eval
- failure condition: test row assigned_split in train/val

## test_manifest_schema.py

- purpose: dataset manifestの必須列を確認する
- fixture: minimal CSV
- expected result: required columns exist
- failure condition: case_id/flair_path/t1_path/mask_path/source_split missing

## test_metrics_golden.py

- purpose: Dice golden casesでmetric定義を固定する
- fixture: perfect match, empty pred, label2-only case
- expected result: perfect match Dice=1.0; label2 ignored
- failure condition: label==2がforeground化される

## test_monai_dataset_smoke.py

- purpose: MONAI Dataset/DataLoaderが1 batchを返すことを確認する
- fixture: temporary tiny NIfTI files or mocked loader
- expected result: image/mask tensors created
- failure condition: shape mismatch or missing modality

## test_training_smoke.py

- purpose: MONAI 3D U-Net smoke training loopが数iteration動くことを確認する
- fixture: tiny synthetic volume
- expected result: loss computed and no NaN
- failure condition: crash, NaN, checkpoint committed to git
