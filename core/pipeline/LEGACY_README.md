# Legacy ISLES-derived Pipeline

This directory contains inherited code from the previous ISLES-oriented skeleton.

For the current WMH2017 MONAI smoke phase, prefer the new lightweight scaffold:

```text
src/wmh2017/
scripts/audit_wmh2017_dataset.py
scripts/audit_wmh2017_labels.py
scripts/make_wmh2017_splits.py
configs/
registry/
tests/
```

Do not edit legacy training/evaluation code until the following pass:

1. label policy tests
2. split leakage tests
3. manifest schema checks
4. metric golden tests
5. MONAI Dataset/DataLoader smoke test
6. MONAI 3D U-Net smoke training

Known risk:
- Some legacy code may still treat non-zero mask as foreground.
- Some legacy names and assumptions come from ISLES.
- Do not use legacy outputs for WMH performance claims without audit.
