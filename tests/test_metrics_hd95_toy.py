"""v4 HD95 toy tests (scipy required at runtime)."""

import numpy as np
import pytest

pytest.importorskip("scipy.ndimage", exc_type=ImportError)

from wmh2017.evaluation.voxel_metrics import hausdorff95_binary, hd95_wmh_label1


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
