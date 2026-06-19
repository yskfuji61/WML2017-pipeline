# Reproducibility rehearsal (v4)

This is **not** production rollback. It verifies prior run evidence, hashes, and report regeneration availability.

Use:

```bash
python scripts/release/run_reproducibility_rehearsal.py --all-scenarios
```

Missing prior artifacts must be marked `MISSING`, not success.
