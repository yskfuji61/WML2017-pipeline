"""T1-R8: default-off FLAIR+T1 multimodal slice stacking for the 2.5D ConvNeXt path.

Pure / monkeypatched unit tests (no training run). Single-modality (default) behavior is verified
unchanged; FLAIR+T1 produces (2*(2k+1), H, W) with the FLAIR block first, T1 block second; center
label and modality-name/path validation are covered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmh2017.data import wmh_slice_dataset as wsd
from wmh2017.data.wmh_slice_dataset import (
    WmhSliceDataset,
    WmhVolumeDataset,
    resolve_modality_keys,
    stack_slices_2_5d,
)
from wmh2017.training import train_convnext_25d as t

OFFSETS = [-2, -1, 0, 1, 2]  # k=2


# --- Test 1: FLAIR-only stack shape unchanged ---
def test_stack_flair_only_shape():
    flair = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    out = stack_slices_2_5d([flair], z=3, offsets=OFFSETS)
    assert out.shape == (5, 8, 8)  # 2k+1
    # center channel (offset 0) equals slice z
    assert np.array_equal(out[2], flair[3])


# --- Test 2: FLAIR+T1 stacked shape is 2*(2k+1) ---
def test_stack_flair_t1_shape():
    flair = np.ones((6, 8, 8), dtype=np.float32)
    t1 = np.full((6, 8, 8), 2.0, dtype=np.float32)
    out = stack_slices_2_5d([flair, t1], z=3, offsets=OFFSETS)
    assert out.shape == (10, 8, 8)  # 2 * (2k+1)


# --- Test 3: FLAIR block precedes T1 block ---
def test_flair_block_precedes_t1_block():
    flair = np.ones((6, 8, 8), dtype=np.float32)
    t1 = np.full((6, 8, 8), 2.0, dtype=np.float32)
    out = stack_slices_2_5d([flair, t1], z=3, offsets=OFFSETS)
    assert np.all(out[0:5] == 1.0)  # FLAIR block first
    assert np.all(out[5:10] == 2.0)  # T1 block second


# --- Test 4: WmhSliceDataset __getitem__ (multimodal + FLAIR-only), center label unchanged ---
class _FakeVol:
    def __init__(self, samples):
        self._s = samples
        self.rows = [{"image": "vol0"}]

    def __len__(self):
        return len(self._s)

    def __getitem__(self, i):
        return self._s[i]


class _FakeMeta:
    def __init__(self, shape):
        self.shape = shape


def test_slice_dataset_getitem_multimodal_and_flaironly(monkeypatch):
    monkeypatch.setattr(wsd, "load_image_metadata", lambda p: _FakeMeta((6, 8, 8)))
    flair = np.ones((6, 8, 8), dtype=np.float32)
    t1 = np.full((6, 8, 8), 2.0, dtype=np.float32)
    label = np.zeros((6, 8, 8), dtype=np.float32)
    label[3, 4, 4] = 1.0  # foreground on slice z=3

    # multimodal
    mm = _FakeVol([{"case_id": "c0", "image": flair, "images": [flair, t1], "label": label}])
    ds_mm = WmhSliceDataset(mm, k=2)
    s = ds_mm[3]  # z=3
    assert tuple(s["image"].shape) == (10, 8, 8)
    assert bool((s["image"][0:5] == 1.0).all()) and bool((s["image"][5:10] == 2.0).all())
    assert tuple(s["label"].shape) == (1, 8, 8)
    assert np.array_equal(s["label"][0].numpy(), label[3])  # center slice label

    # FLAIR-only default (no "images" key)
    so = _FakeVol([{"case_id": "c0", "image": flair, "label": label}])
    ds_so = WmhSliceDataset(so, k=2)
    s2 = ds_so[3]
    assert tuple(s2["image"].shape) == (5, 8, 8)
    assert np.array_equal(s2["label"][0].numpy(), label[3])


# --- Test 5/6: modality-name validation ---
def test_resolve_modality_keys_valid_and_invalid():
    assert resolve_modality_keys(["flair", "t1"]) == ["flair_pre_path", "t1_pre_path"]
    with pytest.raises(ValueError, match="unsupported modality"):
        resolve_modality_keys(["flair", "dwi"])


# --- Test 6: missing modality path raises (tmp CSVs; no image files read in __init__) ---
def test_missing_modality_path_raises(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    split = tmp_path / "split.csv"
    manifest.write_text(
        "case_id,flair_pre_path,t1_pre_path,wmh_path,challenge_split\n" "c0,/x/flair.nii.gz,,/x/wmh.nii.gz,training\n",
        encoding="utf-8",
    )
    split.write_text("case_id,assigned_split\nc0,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing modality path"):
        WmhVolumeDataset(manifest, split, "train", image_keys=["flair_pre_path", "t1_pre_path"])
    # valid multimodal constructs (no file read in __init__)
    manifest.write_text(
        "case_id,flair_pre_path,t1_pre_path,wmh_path,challenge_split\n"
        "c0,/x/flair.nii.gz,/x/t1.nii.gz,/x/wmh.nii.gz,training\n",
        encoding="utf-8",
    )
    ds = WmhVolumeDataset(manifest, split, "train", image_keys=["flair_pre_path", "t1_pre_path"])
    assert ds.multimodal is True
    assert ds.rows[0]["image_paths"] == ["/x/flair.nii.gz", "/x/t1.nii.gz"]


# --- Test 7: trainer resolves modalities + in_channels for FLAIR+T1 ---
def test_trainer_resolves_modalities_and_in_channels():
    assert t.resolve_modalities({}) is None  # default-off
    assert t.resolve_modalities({"modalities": ["flair", "t1"]}) == ["flair", "t1"]
    assert t.resolve_in_channels({}, 2, 1) == 5  # FLAIR-only default
    assert t.resolve_in_channels({}, 2, 2) == 10  # FLAIR+T1
    assert t.resolve_in_channels({"in_channels": 7}, 2, 2) == 7  # explicit override wins


# --- Test 8: no heldout/test path referenced ---
def test_no_heldout_path_in_sources():
    for src in (Path(wsd.__file__), Path(t.__file__)):
        source = src.read_text(encoding="utf-8")
        assert '"heldout_eval"' not in source
        assert '"heldout"' not in source
        assert '"test"' not in source
