from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from wmh2017.inference.export_probabilities import (
    infer_foreground_probability,
    save_case_prediction,
    save_case_probability_map,
)


def test_save_case_probability_map_roundtrip(tmp_path: Path):
    probs = np.array([[[0.1, 0.9], [0.0, 0.5]]], dtype=np.float32)
    out = tmp_path / "case001.npz"
    save_case_probability_map(probs, out)
    loaded = np.load(str(out))["probs"]
    np.testing.assert_allclose(loaded, probs)


def test_infer_foreground_probability_shape(tmp_path: Path):
    image = np.zeros((4, 8, 8), dtype=np.float32)
    image[:, 2:6, 2:6] = 10.0
    image_path = tmp_path / "img.npy"
    np.save(image_path, image)

    model = MagicMock()
    logits = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    logits[:, 1, 2:6, 2:6, 2:6] = 2.0
    model.eval = MagicMock()

    def sliding_window_inference(tensor, roi_size, sw_batch_size, predictor):
        assert predictor is model
        return logits

    monai = {"sliding_window_inference": sliding_window_inference}
    probs = infer_foreground_probability(
        model=model,
        torch=torch,
        monai=monai,
        image_path=str(image_path),
        patch_size=[4, 8, 8],
        device=torch.device("cpu"),
    )
    assert probs.shape == (4, 8, 8)
    assert probs.dtype == np.float32
    assert probs[3, 3, 3] > 0.5


def test_save_case_prediction_writes_binary_and_prob(tmp_path: Path):
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image_path = tmp_path / "img.npy"
    np.save(image_path, image)
    probs = np.full((4, 4, 4), 0.7, dtype=np.float32)
    pred_path = tmp_path / "pred.npy"
    prob_path = tmp_path / "prob.npz"
    save_case_prediction(
        probs=probs,
        threshold=0.5,
        reference_image_path=str(image_path),
        pred_path=pred_path,
        prob_path=prob_path,
    )
    assert pred_path.exists()
    assert prob_path.exists()
