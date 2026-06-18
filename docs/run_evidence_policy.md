# Run Evidence Policy

## Required for each run

- run_id
- run_purpose
- created_at
- git_commit
- config_path
- config_hash
- dataset_manifest_hash
- split_manifest_hash
- model_name
- model_version
- MONAI version
- PyTorch version
- seed
- device
- status
- metric_json_path
- overlay_dir
- failure notes

## Why this matters

Without these fields, results cannot be reproduced, compared, reviewed, or safely promoted later.

## Initial scope

This is not a full production model registry. It is the minimum run evidence needed so that a later model registry or experiment registry can be built without guessing.
