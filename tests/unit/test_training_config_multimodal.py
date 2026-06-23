from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wmh2017.config.training_config import (
    InputModality,
    modalities_to_payload,
    modality_keys,
    resolve_input_modalities,
)
from wmh2017.models.factory import resolve_in_channels

REPO_ROOT = Path(__file__).resolve().parents[2]
A2CV_T1 = REPO_ROOT / "configs" / "experiments" / "cv_t1" / "exp_a2cv_t1_cosine_fold0.yaml"
RC2_T1 = REPO_ROOT / "configs" / "experiments" / "recall_t1" / "exp_rc2_t1_cosine_fold0.yaml"


def _resolve(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg, resolve_input_modalities(cfg["data"])


def test_t1_config_resolves_to_flair_then_t1():
    _, modalities = _resolve(A2CV_T1)
    assert modalities == (
        InputModality(name="flair", manifest_key="flair_pre_path", required=True),
        InputModality(name="t1", manifest_key="t1_pre_path", required=True),
    )
    assert modality_keys(modalities) == ("flair", "t1")


def test_t1_config_in_channels_matches_two():
    cfg, modalities = _resolve(RC2_T1)
    assert cfg["model"]["in_channels"] == 2
    assert resolve_in_channels(model_cfg=cfg["model"], input_modalities=modalities) == 2


def test_in_channels_one_with_two_modalities_raises():
    _, modalities = _resolve(A2CV_T1)
    with pytest.raises(ValueError, match="does not match"):
        resolve_in_channels(model_cfg={"in_channels": 1}, input_modalities=modalities)


def test_modalities_to_payload_roundtrips_names():
    _, modalities = _resolve(A2CV_T1)
    payload = modalities_to_payload(modalities)
    assert payload == [
        {"name": "flair", "manifest_key": "flair_pre_path", "required": True},
        {"name": "t1", "manifest_key": "t1_pre_path", "required": True},
    ]
