# Label Policy — WMH2017

## Required policy

| label | meaning | handling |
|---:|---|---|
| 0 | Background | background |
| 1 | White matter hyperintensities | foreground |
| 2 | Other pathology | ignore |

## Forbidden

```python
foreground = mask > 0
```

This includes label 2 as foreground and corrupts WMH training/evaluation.

## Required

```python
foreground = mask == 1
ignore = mask == 2
```

## Dice handling

Dice for initial smoke is computed on label 1 only.

If label 2 voxels are present in target, they must not become positives. For official-style evaluation, label 2 is ignored according to the challenge policy.

## Unit tests

Required tests:

- label 2 is not foreground
- `mask > 0` gives a different result and is forbidden
- invalid label values outside `{0,1,2}` fail
- Dice with label 2 present still uses label 1 only
