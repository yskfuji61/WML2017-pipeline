"""Empty mask policy via production dice_binary."""

import numpy as np

from wmh2017.evaluation.voxel_metrics import dice_binary


def test_both_empty_masks_score_one():
    pred = np.zeros(4, dtype=int)
    target = np.zeros(4, dtype=int)
    assert dice_binary(pred, target) == 1.0


def test_empty_prediction_nonempty_target_near_zero():
    pred = np.zeros(4, dtype=int)
    target = np.array([0, 1, 0, 0])
    assert dice_binary(pred, target) < 1e-6
