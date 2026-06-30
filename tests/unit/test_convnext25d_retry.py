"""T1-R5: default-off A+B retry enablers for the 2.5D ConvNeXt trainer.

Pure unit tests (no training run) covering the loss dispatch (default tversky_focal / new dice_focal)
and the default-off train-only positive-slice balancing (WeightedRandomSampler). Default/key-absent
behavior is verified to match the prior trainer (TverskyFocalLoss + shuffle DataLoader).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import WeightedRandomSampler

from wmh2017.training import train_convnext_25d as t
from wmh2017.training.losses import DiceFocalLoss, TverskyFocalLoss

SRC = Path(t.__file__)


# --- Test 1: loss resolver default + param passthrough ---
def test_loss_resolver_default_tversky_focal():
    assert isinstance(t.resolve_convnext_loss({}), TverskyFocalLoss)
    # key present without name -> still tversky_focal, reads params
    loss = t.resolve_convnext_loss({"loss": {"alpha": 0.2, "beta": 0.8, "gamma": 2.0}})
    assert isinstance(loss, TverskyFocalLoss)
    assert loss.alpha == 0.2 and loss.beta == 0.8 and loss.gamma == 2.0
    # string form
    assert isinstance(t.resolve_convnext_loss({"loss": "tversky_focal"}), TverskyFocalLoss)


# --- Test 2: dice_focal selectable ---
def test_loss_resolver_dice_focal():
    loss = t.resolve_convnext_loss({"loss": {"name": "dice_focal", "gamma": 2.0}})
    assert isinstance(loss, DiceFocalLoss)
    assert loss.gamma == 2.0
    assert isinstance(t.resolve_convnext_loss({"loss": "dice_focal"}), DiceFocalLoss)


# --- Test 3: invalid loss name raises ---
def test_loss_resolver_invalid_raises():
    with pytest.raises(ValueError, match="training.loss.name"):
        t.resolve_convnext_loss({"loss": {"name": "bogus"}})


# --- Test 6 (helpers): foreground flags + weights ---
class _FakeVol:
    def __init__(self, labels):
        self._labels = labels

    def __getitem__(self, i):
        return {"label": self._labels[int(i)]}


class _FakeSlice:
    def __init__(self):
        # vol0: z0 foreground, z1 empty; vol1: z0 foreground
        self._labels = [
            np.array([[[1.0]], [[0.0]]]),  # shape (2,1,1)
            np.array([[[1.0]]]),  # shape (1,1,1)
        ]
        self.volume_dataset = _FakeVol(self._labels)
        self.index_map = [(0, 0), (0, 1), (1, 0)]


def test_foreground_flags_and_weights():
    ds = _FakeSlice()
    flags = t.compute_slice_foreground_flags(ds)
    assert flags == [True, False, True]
    assert t.positive_slice_sample_weights(flags, 10.0) == [10.0, 1.0, 10.0]
    # foreground slices strictly outweigh background
    w = t.positive_slice_sample_weights(flags, 10.0)
    assert w[0] > w[1] and w[2] > w[1]


# --- Test 4: sampler disabled by default ---
def test_sampler_disabled_by_default():
    loader = t.build_train_loader([0, 1, 2, 3], {})
    assert not isinstance(loader.sampler, WeightedRandomSampler)


# --- Test 5: sampler enabled via config (train-only WeightedRandomSampler) ---
def test_sampler_enabled_when_configured():
    ds = _FakeSlice()
    loader = t.build_train_loader(ds, {"positive_slice_weight": 10.0})
    assert isinstance(loader.sampler, WeightedRandomSampler)
    # weights reflect foreground oversampling
    weights = list(np.asarray(loader.sampler.weights).tolist())
    assert weights == [10.0, 1.0, 10.0]


# --- Test 9: invalid positive_slice_weight raises ---
def test_invalid_positive_slice_weight_raises():
    with pytest.raises(ValueError, match="positive_slice_weight"):
        t.build_train_loader(_FakeSlice(), {"positive_slice_weight": 0})


# --- Test 7: val loader unchanged (train-only balancing) ---
def test_val_loader_unchanged_source():
    source = SRC.read_text(encoding="utf-8")
    # train loader goes through the helper; val loader stays a plain shuffle=False DataLoader.
    assert "train_loader = build_train_loader(train_ds, train_cfg)" in source
    assert "val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)" in source
    # build_train_loader is not applied to the val dataset
    assert "build_train_loader(val_ds" not in source


# --- Test 8: no heldout/test path referenced ---
def test_no_heldout_path_in_source():
    source = SRC.read_text(encoding="utf-8")
    assert "WmhVolumeDataset(" in source
    assert '"train"' in source and '"val"' in source
    assert '"heldout_eval"' not in source
    assert '"heldout"' not in source
    assert '"test"' not in source
