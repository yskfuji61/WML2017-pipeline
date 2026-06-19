"""v4 AVD toy tests (numpy-only production path)."""

import numpy as np

from wmh2017.evaluation.voxel_metrics import absolute_volume_difference_percent, avd_wmh_label1


def test_avd_perfect_match_is_zero():
    mask = np.zeros((3, 3, 3), dtype=int)
    mask[1, 1, 1] = 1
    assert absolute_volume_difference_percent(mask, mask) == 0.0


def test_avd_wmh_label1_ignores_label2_volume():
    pred = np.array([0, 1, 2, 0])
    target = np.array([0, 1, 0, 0])
    assert avd_wmh_label1(pred, target) == 0.0
