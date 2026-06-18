import numpy as np

from wmh2017.data.label_policy import wmh_foreground_mask


def test_wmh_foreground_is_label_1_not_mask_greater_than_zero():
    mask = np.array([0, 1, 2])
    foreground = wmh_foreground_mask(mask)
    assert foreground.tolist() == [False, True, False]
