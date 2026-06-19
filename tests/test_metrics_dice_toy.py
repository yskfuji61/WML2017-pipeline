"""v4 dice toy tests against production dice implementation."""

import numpy as np
import pytest

from wmh2017.evaluation.voxel_metrics import dice_binary, dice_wmh_label1


def test_dice_perfect_match():
    a = np.array([0, 1, 1, 0])
    assert dice_binary(a, a) == pytest.approx(1.0)


def test_dice_empty_prediction_target_nonempty_near_zero():
    pred = np.array([0, 0, 0, 0])
    target = np.array([0, 1, 0, 0])
    assert dice_binary(pred, target) < 1e-6


def test_dice_wmh_label1_ignores_label2_as_foreground():
    pred = np.array([0, 2, 0, 0])
    target = np.array([0, 1, 0, 0])
    assert dice_wmh_label1(pred, target) < 1e-6
