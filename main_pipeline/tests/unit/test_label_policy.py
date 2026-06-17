import numpy as np
import pytest

from wmh2017.data.label_policy import (
    audit_mask,
    validate_label_values,
    wmh_foreground_mask,
    wmh_ignore_mask,
)


def test_label2_is_not_foreground():
    mask = np.array([[0, 1, 2], [2, 1, 0]])
    fg = wmh_foreground_mask(mask)
    ig = wmh_ignore_mask(mask)
    assert fg.sum() == 2
    assert ig.sum() == 2
    assert not np.any(fg & ig)


def test_mask_greater_than_zero_would_be_wrong():
    mask = np.array([0, 1, 2])
    assert (mask > 0).sum() == 2
    assert wmh_foreground_mask(mask).sum() == 1


def test_invalid_label_values_fail():
    mask = np.array([0, 1, 3])
    with pytest.raises(ValueError):
        validate_label_values(mask)


def test_audit_mask():
    mask = np.array([0, 1, 2, 2])
    audit = audit_mask(mask)
    assert audit.values == (0, 1, 2)
    assert audit.has_label2 is True
    assert audit.foreground_voxels == 1
    assert audit.ignore_voxels == 2
