import numpy as np


def test_dice_symmetry_property():
    pred = np.array([0, 1, 1, 0, 0])
    target = np.array([0, 1, 0, 1, 0])

    def dice(a, b):
        a = a.astype(bool)
        b = b.astype(bool)
        inter = np.logical_and(a, b).sum()
        return float(2 * inter / (a.sum() + b.sum() + 1e-8))

    assert dice(pred, target) == dice(target, pred)
