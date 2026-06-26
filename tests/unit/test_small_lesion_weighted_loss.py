"""Default-off small-lesion-weighted CE loss: helper + factory + numeric behavior."""

from __future__ import annotations

import numpy as np
import pytest

from wmh2017.training.loss_factory import build_loss
from wmh2017.training.losses import MonaiDiceCELossWrapper
from wmh2017.training.small_lesion_sampling import small_lesion_weight_map


class _StubDiceCE:
    """Stand-in for monai DiceCELoss so factory tests need no real monai."""

    def __init__(self, **kw):
        self.kw = kw


_STUB_MONAI = {"DiceCELoss": _StubDiceCE}


def _label_small_and_large() -> np.ndarray:
    lab = np.zeros((20, 20, 20), dtype=np.int64)
    lab[1, 1, 1] = 1
    lab[1, 1, 2] = 1  # 2-voxel small component
    lab[10:15, 10:15, 10:14] = 1  # 100-voxel large component
    return lab


# --- weight-map helper (pure numpy) ----------------------------------------
def test_weight_map_correctness() -> None:
    lab = _label_small_and_large()
    wmap = small_lesion_weight_map(lab, max_voxels=10, weight=5.0)
    assert wmap.dtype == np.float32
    assert wmap[1, 1, 1] == 5.0 and wmap[1, 1, 2] == 5.0  # small voxels weighted
    assert wmap[12, 12, 12] == 1.0  # large-component voxel not weighted
    assert int((wmap == 5.0).sum()) == 2  # exactly the 2 small voxels


def test_weight_map_no_small_is_all_ones() -> None:
    lab = np.zeros((20, 20, 20), dtype=np.int64)
    lab[5:10, 5:10, 5:10] = 1  # 125-voxel large only
    wmap = small_lesion_weight_map(lab, max_voxels=10, weight=5.0)
    assert np.all(wmap == 1.0)


def test_weight_map_weight_one_is_noop() -> None:
    lab = _label_small_and_large()
    assert np.all(small_lesion_weight_map(lab, max_voxels=10, weight=1.0) == 1.0)


# --- factory default-off / validation (no real monai needed) ----------------
def test_factory_default_off_returns_dice_ce() -> None:
    assert isinstance(build_loss({"loss": "dice_ce"}, _STUB_MONAI), MonaiDiceCELossWrapper)
    assert isinstance(build_loss({"loss": {"name": "dice_ce"}}, _STUB_MONAI), MonaiDiceCELossWrapper)
    assert isinstance(
        build_loss({"loss": {"name": "dice_ce", "small_lesion_ce_weight": 1.0}}, _STUB_MONAI),
        MonaiDiceCELossWrapper,
    )


def test_factory_enabled_returns_weighted_loss() -> None:
    from wmh2017.training.losses import SmallLesionWeightedDiceCELoss

    loss = build_loss(
        {"loss": {"name": "dice_ce", "small_lesion_ce_weight": 5.0, "small_lesion_max_voxels": 10}},
        _STUB_MONAI,
    )
    assert isinstance(loss, SmallLesionWeightedDiceCELoss)
    assert loss.small_lesion_ce_weight == 5.0 and loss.small_lesion_max_voxels == 10


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="small_lesion_ce_weight"):
        build_loss({"loss": {"name": "dice_ce", "small_lesion_ce_weight": 0.5}}, _STUB_MONAI)
    with pytest.raises(ValueError, match="small_lesion_max_voxels"):
        build_loss({"loss": {"name": "dice_ce", "small_lesion_max_voxels": 0}}, _STUB_MONAI)


# --- numeric behavior (torch) ----------------------------------------------
def _logits_for(lab: np.ndarray, torch, small_fn: bool):
    """2-class logits that predict GT well, optionally with a small-lesion false negative."""
    import numpy as _np

    fg = lab > 0
    bg_logit = _np.where(fg, -4.0, 4.0).astype(_np.float32)  # ch0 (background)
    fg_logit = _np.where(fg, 4.0, -4.0).astype(_np.float32)  # ch1 (foreground)
    if small_fn:
        # force a false negative on the 2-voxel small lesion (predict background there)
        for z, y, x in [(1, 1, 1), (1, 1, 2)]:
            bg_logit[z, y, x] = 4.0
            fg_logit[z, y, x] = -4.0
    logits = _np.stack([bg_logit, fg_logit])[None]  # (1, 2, Z, Y, X)
    return torch.tensor(logits)


def test_weighted_ce_increases_with_weight_on_small_fn() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.losses import SmallLesionWeightedDiceCELoss

    lab = _label_small_and_large()
    logits = _logits_for(lab, torch, small_fn=True)
    targets = torch.tensor(lab[None, None])  # (1,1,Z,Y,X)
    loss_w1 = SmallLesionWeightedDiceCELoss(1.0, 10)(logits, targets)
    loss_w5 = SmallLesionWeightedDiceCELoss(5.0, 10)(logits, targets)
    assert torch.isfinite(loss_w1) and torch.isfinite(loss_w5)
    assert float(loss_w5) > float(loss_w1)  # up-weighting the small FN raises the loss


def test_no_small_lesion_weight_is_neutral() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.losses import SmallLesionWeightedDiceCELoss

    lab = np.zeros((16, 16, 16), dtype=np.int64)
    lab[4:9, 4:9, 4:9] = 1  # large only
    logits = _logits_for(lab, torch, small_fn=False)
    targets = torch.tensor(lab[None, None])
    l1 = SmallLesionWeightedDiceCELoss(1.0, 10)(logits, targets)
    l5 = SmallLesionWeightedDiceCELoss(5.0, 10)(logits, targets)
    assert abs(float(l1) - float(l5)) < 1e-6  # no small lesion -> weight irrelevant


def test_shape_dtype_contract() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.losses import SmallLesionWeightedDiceCELoss

    lab = _label_small_and_large()
    logits = _logits_for(lab, torch, small_fn=False)
    out = SmallLesionWeightedDiceCELoss(5.0, 10)(logits, torch.tensor(lab[None, None]))
    assert out.ndim == 0 and torch.isfinite(out)
