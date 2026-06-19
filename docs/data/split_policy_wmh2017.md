# WMH2017 split policy (v4)

- Train/val split uses challenge **training** cases only (`challenge_split=training`).
- Test/hidden-test cases are held out (`heldout_eval`) and must not appear in train or val.
- Default seed: `20260616` for v4 smoke alignment.
- Split overlap verification: `scripts/data/verify_no_split_overlap.py`.
