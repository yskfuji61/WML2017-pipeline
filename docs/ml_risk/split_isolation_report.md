# Split isolation report

## Conclusion

Train/validation splits are created only from challenge `training` cases. Challenge `test` cases are assigned `heldout_eval` and blocked from training, validation, and model selection.

## Evidence

- Policy: [docs/split_policy.md](../split_policy.md)
- Implementation: `src/wmh2017/data/splits.py`
- Tests: `tests/unit/test_split_no_leakage.py`, `tests/contract/test_no_test_split_for_training.py`

## Verified behaviors

| Check | Status |
|-------|--------|
| Test rows never assigned train/val | PASS (unit tests) |
| Unknown source split blocked | PASS (unit tests) |
| Training without primary WMH excluded | PASS (unit tests) |

## Blocked claims

Clinical deployment, leaderboard/SOTA equivalence, and external validity beyond documented smoke scope remain blocked.
