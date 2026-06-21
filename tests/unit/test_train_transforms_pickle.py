from __future__ import annotations

import pickle

import numpy as np

from wmh2017.training.transforms import _label_to_foreground_mask


def test_label_to_foreground_mask_maps_label1_only() -> None:
    label = np.array([0, 1, 2, 1], dtype=np.int64)
    out = _label_to_foreground_mask(label)
    np.testing.assert_array_equal(out, np.array([0, 1, 0, 1], dtype=np.int64))


def test_label_to_foreground_mask_is_pickleable_for_dataloader_workers() -> None:
    payload = pickle.dumps(_label_to_foreground_mask)
    restored = pickle.loads(payload)
    assert restored(np.array([1])).tolist() == [1]
