"""Shared guard for the ``challenge_split=test`` boundary.

Single source of truth for refusing test-split cases. By default any case whose manifest
``challenge_split == "test"`` is rejected (training, validation, threshold tuning, ensemble,
slice datasets). A narrow, explicit, opt-in override permits such a case **only** for
released-label *local* test inference/evaluation, and **only** when ``assigned_split ==
"heldout_eval"``. The override never applies to train/val/tuning/selection and is off by default,
so absent the flag behavior is identical to the prior per-module checks.

This is local validation only; the override does not make any result official/leaderboard/SOTA/
clinical comparable.
"""

from __future__ import annotations

RELEASED_LABEL_LOCAL_TEST = "released_label_local_test"
_HELDOUT_EVAL = "heldout_eval"


def is_released_label_local_test_case(
    challenge_split: str,
    assigned_split: str,
    *,
    allow_released_label_local_test: bool,
) -> bool:
    """True iff the explicit override applies: flag on, challenge_split=test, heldout_eval."""
    return (
        bool(allow_released_label_local_test)
        and str(challenge_split).strip().lower() == "test"
        and str(assigned_split).strip().lower() == _HELDOUT_EVAL
    )


def guard_challenge_split_test(
    case_id: str,
    challenge_split: str,
    *,
    assigned_split: str,
    allow_released_label_local_test: bool = False,
    context: str = "",
) -> None:
    """Raise if a ``challenge_split=test`` case is used outside the released-label override.

    Non-test cases pass. A test case passes only when the explicit released-label local-test
    override is enabled AND ``assigned_split == "heldout_eval"``. Default-off: with the flag absent
    any test case raises, and the message always contains ``"challenge_split=test"``.
    """
    if str(challenge_split).strip().lower() != "test":
        return
    if is_released_label_local_test_case(
        challenge_split,
        assigned_split,
        allow_released_label_local_test=allow_released_label_local_test,
    ):
        return
    suffix = f" [{context}]" if context else ""
    raise ValueError(f"case_id={case_id} belongs to challenge_split=test; must not be used here{suffix}")
