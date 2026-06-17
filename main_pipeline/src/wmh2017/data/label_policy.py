"""WMH2017 label policy.

Policy:
- 0: background
- 1: WMH foreground
- 2: ignore / other pathology

Do not use `mask > 0` as foreground.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import numpy as np

ALLOWED_LABELS: Set[int] = {0, 1, 2}
FOREGROUND_LABEL = 1
IGNORE_LABEL = 2


@dataclass(frozen=True)
class LabelAudit:
    values: tuple[int, ...]
    has_label2: bool
    foreground_voxels: int
    ignore_voxels: int


def unique_label_values(mask: np.ndarray) -> tuple[int, ...]:
    """Return sorted integer label values from a mask."""
    values = np.unique(mask)
    return tuple(int(v) for v in values.tolist())


def validate_label_values(mask: np.ndarray, allowed: Iterable[int] = ALLOWED_LABELS) -> tuple[int, ...]:
    """Validate that mask values are within the WMH2017 label set."""
    values = unique_label_values(mask)
    allowed_set = {int(v) for v in allowed}
    unexpected = sorted(set(values) - allowed_set)
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}; allowed={sorted(allowed_set)}")
    return values


def wmh_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """Return binary foreground mask for WMH lesions only."""
    validate_label_values(mask)
    return np.asarray(mask) == FOREGROUND_LABEL


def wmh_ignore_mask(mask: np.ndarray) -> np.ndarray:
    """Return binary ignore mask for label 2."""
    validate_label_values(mask)
    return np.asarray(mask) == IGNORE_LABEL


def audit_mask(mask: np.ndarray) -> LabelAudit:
    """Summarize label values and policy-relevant voxel counts."""
    values = validate_label_values(mask)
    fg = wmh_foreground_mask(mask)
    ig = wmh_ignore_mask(mask)
    return LabelAudit(
        values=values,
        has_label2=IGNORE_LABEL in values,
        foreground_voxels=int(fg.sum()),
        ignore_voxels=int(ig.sum()),
    )
