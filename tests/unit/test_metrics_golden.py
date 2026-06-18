import numpy as np

from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_binary, lesion_recall_f1_wmh_label1
from wmh2017.evaluation.voxel_metrics import (
    absolute_volume_difference_percent,
    avd_wmh_label1,
    dice_binary,
    dice_wmh_label1,
    hausdorff95_binary,
    hd95_wmh_label1,
)


def test_dice_perfect_match():
    a = np.array([0, 1, 1, 0])
    assert dice_binary(a, a) == 1.0


def test_dice_empty_prediction_target_nonempty_near_zero():
    pred = np.array([0, 0, 0, 0])
    target = np.array([0, 1, 0, 0])
    assert dice_binary(pred, target) < 1e-6


def test_dice_wmh_label1_ignores_label2_as_foreground():
    pred = np.array([0, 2, 0, 0])
    target = np.array([0, 1, 0, 0])
    assert dice_wmh_label1(pred, target) < 1e-6


def test_dice_wmh_label1_perfect_with_label2_present():
    pred = np.array([0, 1, 2, 0])
    target = np.array([0, 1, 2, 0])
    assert dice_wmh_label1(pred, target) == 1.0


def test_avd_perfect_match_is_zero():
    mask = np.zeros((3, 3, 3), dtype=int)
    mask[1, 1, 1] = 1
    assert absolute_volume_difference_percent(mask, mask) == 0.0


def test_avd_wmh_label1_ignores_label2_volume():
    pred = np.array([0, 1, 2, 0])
    target = np.array([0, 1, 0, 0])
    assert avd_wmh_label1(pred, target) == 0.0


def test_hd95_perfect_match_is_zero():
    mask = np.zeros((5, 5, 5), dtype=int)
    mask[2, 2, 2] = 1
    assert hausdorff95_binary(mask, mask) == 0.0


def test_hd95_wmh_label1_ignores_label2():
    pred = np.zeros((5, 5, 5), dtype=int)
    target = np.zeros((5, 5, 5), dtype=int)
    pred[2, 2, 2] = 1
    pred[1, 1, 1] = 2
    target[2, 2, 2] = 1
    assert hd95_wmh_label1(pred, target) == 0.0


def test_lesion_recall_f1_perfect_match():
    pred = np.zeros((6, 6, 6), dtype=bool)
    target = np.zeros((6, 6, 6), dtype=bool)
    pred[1, 1, 1] = True
    target[1, 1, 1] = True
    out = lesion_recall_f1_binary(pred, target)
    assert out["lesion_recall"] == 1.0
    assert out["lesion_f1"] == 1.0


def test_lesion_recall_detects_false_negative():
    pred = np.zeros((6, 6, 6), dtype=bool)
    target = np.zeros((6, 6, 6), dtype=bool)
    target[1, 1, 1] = True
    target[4, 4, 4] = True
    pred[1, 1, 1] = True
    out = lesion_recall_f1_binary(pred, target, connectivity=6)
    assert out["lesion_recall"] == 0.5
    assert out["lesion_f1"] < 1.0


def test_lesion_wmh_label1_ignores_label2_prediction():
    pred = np.zeros((6, 6, 6), dtype=int)
    target = np.zeros((6, 6, 6), dtype=int)
    pred[1, 1, 1] = 2
    target[1, 1, 1] = 1
    out = lesion_recall_f1_wmh_label1(pred, target)
    assert out["lesion_recall"] == 0.0
    assert out["lesion_f1"] == 0.0
