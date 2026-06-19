from __future__ import annotations

from wmh2017.training.train_monai import _amp_policy


def test_amp_policy_cuda_when_requested() -> None:
    enabled, device = _amp_policy(True, "cuda")
    assert enabled is True
    assert device == "cuda"


def test_amp_policy_disabled_on_mps_even_when_requested() -> None:
    enabled, device = _amp_policy(True, "mps")
    assert enabled is False
    assert device is None


def test_amp_policy_disabled_when_not_requested() -> None:
    enabled, device = _amp_policy(False, "cuda")
    assert enabled is False
    assert device is None
