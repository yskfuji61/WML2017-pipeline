from __future__ import annotations

import pytest

from wmh2017.config.training_config import InputModality
from wmh2017.models.factory import build_unet, resolve_in_channels


class _FakeUNet:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _modalities(n: int) -> tuple[InputModality, ...]:
    return tuple(InputModality(name=f"m{i}", manifest_key=f"k{i}") for i in range(n))


def test_resolve_infers_channel_count_when_unset():
    assert resolve_in_channels(model_cfg={}, input_modalities=_modalities(2)) == 2


def test_resolve_accepts_matching_in_channels():
    assert resolve_in_channels(model_cfg={"in_channels": 1}, input_modalities=_modalities(1)) == 1


def test_resolve_rejects_mismatched_in_channels():
    with pytest.raises(ValueError, match="does not match"):
        resolve_in_channels(model_cfg={"in_channels": 1}, input_modalities=_modalities(2))


def test_build_unet_legacy_defaults_to_single_channel():
    monai = {"UNet": _FakeUNet}
    model = build_unet(monai, {"model": {"spatial_dims": 3, "out_channels": 2}})
    assert model.kwargs["in_channels"] == 1


def test_build_unet_validates_against_modalities():
    monai = {"UNet": _FakeUNet}
    with pytest.raises(ValueError, match="does not match"):
        build_unet(monai, {"model": {"in_channels": 1}}, input_modalities=_modalities(2))


def test_build_unet_infers_channels_from_modalities():
    monai = {"UNet": _FakeUNet}
    model = build_unet(monai, {"model": {}}, input_modalities=_modalities(2))
    assert model.kwargs["in_channels"] == 2
