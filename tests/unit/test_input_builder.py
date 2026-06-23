from __future__ import annotations

from pathlib import Path

import numpy as np

from wmh2017.inference.input_builder import load_normalized_input_volume, to_batched_tensor


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def to(self, device: object) -> _FakeTensor:
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(arr: np.ndarray) -> _FakeTensor:
        return _FakeTensor(arr)


def _save(path: Path, arr: np.ndarray) -> str:
    np.save(path, arr)
    return str(path)


def test_single_channel_volume_shape(tmp_path: Path):
    img = _save(tmp_path / "flair.npy", np.ones((4, 8, 8), dtype=np.float32))
    volume = load_normalized_input_volume(image_paths={"image": img}, input_keys=("image",))
    assert volume.shape == (1, 4, 8, 8)
    assert volume.dtype == np.float32


def test_two_channel_volume_shape(tmp_path: Path):
    flair = _save(tmp_path / "flair.npy", np.ones((4, 8, 8), dtype=np.float32))
    t1 = _save(tmp_path / "t1.npy", np.ones((4, 8, 8), dtype=np.float32) * 2)
    volume = load_normalized_input_volume(image_paths={"flair": flair, "t1": t1}, input_keys=("flair", "t1"))
    assert volume.shape == (2, 4, 8, 8)


def test_to_batched_tensor_single_channel(tmp_path: Path):
    img = _save(tmp_path / "flair.npy", np.ones((4, 8, 8), dtype=np.float32))
    volume = load_normalized_input_volume(image_paths={"image": img}, input_keys=("image",))
    tensor = to_batched_tensor(_FakeTorch(), volume, device="cpu")
    assert tensor.arr.shape == (1, 1, 4, 8, 8)


def test_to_batched_tensor_two_channel(tmp_path: Path):
    flair = _save(tmp_path / "flair.npy", np.ones((4, 8, 8), dtype=np.float32))
    t1 = _save(tmp_path / "t1.npy", np.ones((4, 8, 8), dtype=np.float32))
    volume = load_normalized_input_volume(image_paths={"flair": flair, "t1": t1}, input_keys=("flair", "t1"))
    tensor = to_batched_tensor(_FakeTorch(), volume, device="cpu")
    assert tensor.arr.shape == (1, 2, 4, 8, 8)
