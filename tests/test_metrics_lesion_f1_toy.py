"""v4 lesion F1 toy tests (scipy required at runtime)."""

import numpy as np
import pytest

pytest.importorskip("scipy.ndimage", exc_type=ImportError)

from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_binary


def test_lesion_recall_f1_perfect_match():
    pred = np.zeros((6, 6, 6), dtype=bool)
    target = np.zeros((6, 6, 6), dtype=bool)
    pred[1, 1, 1] = True
    target[1, 1, 1] = True
    out = lesion_recall_f1_binary(pred, target)
    assert out["lesion_recall"] == 1.0
    assert out["lesion_f1"] == 1.0
