from __future__ import annotations

from wmh2017.training.train_monai import _amp_policy


def test_amp_policy_cuda_when_requested() -> None:
    enabled, device, policy = _amp_policy(True, "cuda")
    assert enabled is True
    assert device == "cuda"
    assert policy == "cuda_amp_optional"


def test_amp_policy_mps_keeps_float32_for_accuracy() -> None:
    enabled, device, policy = _amp_policy(True, "mps")
    assert enabled is False
    assert device is None
    assert policy == "mps_float32_accuracy_first"


def test_amp_policy_disabled_when_not_requested() -> None:
    enabled, device, policy = _amp_policy(False, "cuda")
    assert enabled is False
    assert device is None
    assert policy == "float32_full_precision"
