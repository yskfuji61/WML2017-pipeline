"""T1-R8x: default-off multimodal support in the 2.5D ConvNeXt probability export path.

Fake-model / tmp-CSV unit tests (no checkpoint, no nii, no training). Verifies the export input stack
is FLAIR-only by default (2k+1) and FLAIR+T1 when opted in (2*(2k+1), FLAIR block first), plus modality
name/path validation. The exported npz schema is unchanged (save_case_probability_map untouched).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wmh2017.data.wmh_slice_dataset import WmhVolumeDataset, resolve_modality_keys
from wmh2017.inference import export_convnext_probabilities as exp
from wmh2017.inference.export_convnext_probabilities import infer_volume_probabilities

OFFSETS = [-2, -1, 0, 1, 2]  # k=2
DEV = torch.device("cpu")


class _CaptureModel:
    """Records input channel count / values; returns dummy (B,1,H,W) logits."""

    def __init__(self):
        self.shapes: list[tuple[int, ...]] = []
        self.first_input: torch.Tensor | None = None

    def eval(self):
        return self

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(tensor.shape))
        if self.first_input is None:
            self.first_input = tensor.detach().clone()
        b, _c, h, w = tensor.shape
        return torch.zeros((b, 1, h, w))


# --- Test 1: default FLAIR-only export input stack unchanged (2k+1) ---
def test_export_flair_only_input_channels():
    flair = np.ones((6, 8, 8), dtype=np.float32)
    m = _CaptureModel()
    out = infer_volume_probabilities(model=m, volumes=[flair], offsets=OFFSETS, device=DEV)
    assert out.shape == (6, 8, 8)  # (Z,H,W) preserved
    assert m.shapes[0][1] == 5  # 2k+1 input channels


# --- Test 2: FLAIR+T1 export input stack = 2*(2k+1) ---
def test_export_flair_t1_input_channels():
    flair = np.ones((6, 8, 8), dtype=np.float32)
    t1 = np.full((6, 8, 8), 2.0, dtype=np.float32)
    m = _CaptureModel()
    infer_volume_probabilities(model=m, volumes=[flair, t1], offsets=OFFSETS, device=DEV)
    assert m.shapes[0][1] == 10  # 2 * (2k+1)


# --- Test 3: FLAIR block precedes T1 block ---
def test_export_flair_block_precedes_t1():
    flair = np.ones((6, 8, 8), dtype=np.float32)
    t1 = np.full((6, 8, 8), 2.0, dtype=np.float32)
    m = _CaptureModel()
    infer_volume_probabilities(model=m, volumes=[flair, t1], offsets=OFFSETS, device=DEV)
    inp = m.first_input[0]  # (10,H,W)
    assert bool((inp[0:5] == 1.0).all())  # FLAIR block
    assert bool((inp[5:10] == 2.0).all())  # T1 block


# --- back-compat: image3d= still works and matches volumes=[image3d] ---
def test_infer_backcompat_image3d():
    flair = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    m1, m2 = _CaptureModel(), _CaptureModel()
    o1 = infer_volume_probabilities(model=m1, image3d=flair, offsets=OFFSETS, device=DEV)
    o2 = infer_volume_probabilities(model=m2, volumes=[flair], offsets=OFFSETS, device=DEV)
    assert o1.shape == o2.shape == (6, 8, 8)
    assert m1.shapes[0][1] == m2.shapes[0][1] == 5


# --- Test 4: invalid modality raises (export uses resolve_modality_keys) ---
def test_invalid_modality_raises():
    with pytest.raises(ValueError, match="unsupported modality"):
        resolve_modality_keys(["flair", "dwi"])


# --- Test 5: missing modality path raises (tmp CSVs; no image files) ---
def test_missing_modality_path_raises(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    split = tmp_path / "split.csv"
    manifest.write_text(
        "case_id,flair_pre_path,t1_pre_path,wmh_path,challenge_split\nc0,/x/flair.nii.gz,,/x/wmh.nii.gz,training\n",
        encoding="utf-8",
    )
    split.write_text("case_id,assigned_split\nc0,val\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing modality path"):
        WmhVolumeDataset(manifest, split, "val", image_keys=["flair_pre_path", "t1_pre_path"])


# --- Test 6: npz schema unchanged (export still uses save_case_probability_map + {case_id}.npz) ---
def test_export_npz_schema_unchanged_source():
    source = Path(exp.__file__).read_text(encoding="utf-8")
    assert "save_case_probability_map(probs, prob_path)" in source
    assert "f\"{sample['case_id']}.npz\"" in source
    # save_case_probability_map is imported, not redefined here.
    assert "def save_case_probability_map" not in source
