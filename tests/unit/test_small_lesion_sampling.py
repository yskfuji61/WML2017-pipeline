"""Default-off small-lesion-aware sampling: pure helpers + wiring + crop-shape contract."""

from __future__ import annotations

import numpy as np
import pytest

from wmh2017.training.small_lesion_sampling import (
    SmallLesionFgBgIndicesd,
    build_biased_fg_indices,
    resolve_small_lesion_sampling_cfg,
    small_lesion_flat_indices,
)


def _label_with_small_and_large() -> np.ndarray:
    """3D label: one 2-voxel component + one 100-voxel (5x5x4) blob, well separated."""
    lab = np.zeros((20, 20, 20), dtype=np.uint8)
    lab[1, 1, 1] = 1
    lab[1, 1, 2] = 1  # 2-voxel small component
    lab[10:15, 10:15, 10:14] = 1  # 5*5*4 = 100-voxel large component
    return lab


def test_default_off_resolves_disabled() -> None:
    assert resolve_small_lesion_sampling_cfg({}) == (False, 0.0, 10)
    assert resolve_small_lesion_sampling_cfg({"small_lesion_center_prob": 0.0}) == (False, 0.0, 10)


def test_default_off_wiring_absent_then_present() -> None:
    # Stub monai: each op factory returns a labeled sentinel so we can inspect the op list.
    from wmh2017.training.transforms import build_monai_transforms

    class _Op:
        def __init__(self, name, **kw):
            self.name = name
            self.kw = kw

    def _stub(name):
        return lambda **kw: _Op(name, **kw)

    monai = {
        "LoadImaged": _stub("LoadImaged"),
        "EnsureChannelFirstd": _stub("EnsureChannelFirstd"),
        "Lambdad": _stub("Lambdad"),
        "RandCropByPosNegLabeld": _stub("RandCropByPosNegLabeld"),
        "ResizeWithPadOrCropd": _stub("ResizeWithPadOrCropd"),
        "EnsureTyped": _stub("EnsureTyped"),
        "Compose": lambda ops: ops,  # return the op list directly
    }
    # disabled: no SmallLesionFgBgIndicesd; crop has no fg/bg index keys
    ops_off = build_monai_transforms(monai, [8, 8, 8], train=True, train_cfg={})
    assert not any(isinstance(o, SmallLesionFgBgIndicesd) for o in ops_off)
    crop_off = next(o for o in ops_off if getattr(o, "name", "") == "RandCropByPosNegLabeld")
    assert "fg_indices_key" not in crop_off.kw and "bg_indices_key" not in crop_off.kw
    # enabled: SmallLesionFgBgIndicesd present; crop wired to the index keys
    ops_on = build_monai_transforms(
        monai,
        [8, 8, 8],
        train=True,
        train_cfg={"sampling": {"small_lesion_center_prob": 0.5, "small_lesion_max_voxels": 10}},
    )
    assert any(isinstance(o, SmallLesionFgBgIndicesd) for o in ops_on)
    crop_on = next(o for o in ops_on if getattr(o, "name", "") == "RandCropByPosNegLabeld")
    assert crop_on.kw["fg_indices_key"] == "fg_indices"
    assert crop_on.kw["bg_indices_key"] == "bg_indices"


def test_small_component_detection() -> None:
    lab = _label_with_small_and_large()
    small = small_lesion_flat_indices(lab > 0, max_voxels=10)
    expected = np.flatnonzero((lab > 0).ravel())  # all fg
    small_set = set(small.tolist())
    # exactly the 2 small-component voxels, none of the 100 large ones
    assert len(small_set) == 2
    assert small_set.issubset(set(expected.tolist()))
    # the large blob voxels are excluded
    large_idx = np.flatnonzero(((lab > 0) & ~_only_small_mask(lab)).ravel())
    assert small_set.isdisjoint(set(large_idx.tolist()))


def _only_small_mask(lab: np.ndarray) -> np.ndarray:
    m = np.zeros_like(lab, dtype=bool)
    m[1, 1, 1] = True
    m[1, 1, 2] = True
    return m


def test_prob_one_selects_small_only() -> None:
    lab = _label_with_small_and_large()
    fg = np.flatnonzero((lab > 0).ravel()).astype(np.int64)
    small = small_lesion_flat_indices(lab > 0, max_voxels=10)
    biased = build_biased_fg_indices(fg, small, 1.0)
    assert set(biased.tolist()) == set(small.tolist())  # positives always centered on small


def test_fallback_when_no_small_lesion() -> None:
    lab = np.zeros((20, 20, 20), dtype=np.uint8)
    lab[5:10, 5:10, 5:10] = 1  # 125-voxel large only
    fg = np.flatnonzero((lab > 0).ravel()).astype(np.int64)
    small = small_lesion_flat_indices(lab > 0, max_voxels=10)
    assert small.size == 0
    biased = build_biased_fg_indices(fg, small, 0.5)
    assert np.array_equal(biased, fg)  # unchanged fallback


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="small_lesion_center_prob"):
        resolve_small_lesion_sampling_cfg({"small_lesion_center_prob": 1.5})
    with pytest.raises(ValueError, match="small_lesion_max_voxels"):
        resolve_small_lesion_sampling_cfg({"small_lesion_max_voxels": 0})


def test_indices_transform_writes_biased_fg_and_bg() -> None:
    lab = _label_with_small_and_large()
    image = (lab > 0).astype(np.float32)[None]  # (1, D, H, W); brain = fg here
    label = lab.astype(np.int64)[None]
    t = SmallLesionFgBgIndicesd(max_voxels=10, small_center_prob=1.0)
    out = t({"image": image, "label": label})
    small = small_lesion_flat_indices(lab > 0, max_voxels=10)
    assert set(out["fg_indices"].tolist()) == set(small.tolist())
    assert out["fg_indices"].dtype == np.int64 and out["bg_indices"].dtype == np.int64


def test_crop_shape_contract() -> None:
    pytest.importorskip("monai")
    pytest.importorskip("torch")
    from monai.transforms import Compose, RandCropByPosNegLabeld, ResizeWithPadOrCropd

    lab = _label_with_small_and_large()
    image = np.random.RandomState(0).randn(1, 20, 20, 20).astype(np.float32)
    label = lab.astype(np.int64)[None]
    patch = (8, 8, 8)
    pipe = Compose(
        [
            SmallLesionFgBgIndicesd(max_voxels=10, small_center_prob=1.0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
                fg_indices_key="fg_indices",
                bg_indices_key="bg_indices",
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch),
        ]
    )
    out = pipe({"image": image, "label": label})
    assert isinstance(out, list) and len(out) == 2
    for d in out:
        assert tuple(d["image"].shape[1:]) == patch
        assert tuple(d["label"].shape[1:]) == patch
