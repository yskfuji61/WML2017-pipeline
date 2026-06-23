from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from wmh2017.inference.export_probabilities import infer_foreground_probability


def test_two_channel_input_builds_2ch_tensor(tmp_path: Path):
    flair = np.zeros((4, 8, 8), dtype=np.float32)
    flair[:, 2:6, 2:6] = 10.0
    t1 = np.zeros((4, 8, 8), dtype=np.float32)
    t1[:, 2:6, 2:6] = 5.0
    flair_path = tmp_path / "flair.npy"
    t1_path = tmp_path / "t1.npy"
    np.save(flair_path, flair)
    np.save(t1_path, t1)

    model = MagicMock()
    model.eval = MagicMock()
    logits = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    logits[:, 1, 2:6, 2:6, 2:6] = 2.0
    seen = {}

    def sliding_window_inference(tensor, roi_size, sw_batch_size, predictor):
        assert predictor is model
        seen["shape"] = tuple(tensor.shape)
        return logits

    probs = infer_foreground_probability(
        model=model,
        torch=torch,
        monai={"sliding_window_inference": sliding_window_inference},
        image_paths={"flair": str(flair_path), "t1": str(t1_path)},
        input_keys=("flair", "t1"),
        patch_size=[4, 8, 8],
        device=torch.device("cpu"),
    )
    assert seen["shape"] == (1, 2, 4, 8, 8)
    assert probs.shape == (4, 8, 8)
    assert probs.dtype == np.float32


def test_requires_an_image_source():
    import pytest

    with pytest.raises(ValueError, match="image_path or image_paths"):
        infer_foreground_probability(
            model=MagicMock(),
            torch=torch,
            monai={"sliding_window_inference": lambda **kw: None},
            patch_size=[4, 4, 4],
            device=torch.device("cpu"),
        )
