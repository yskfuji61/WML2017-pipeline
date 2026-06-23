from __future__ import annotations

from pathlib import Path

import yaml

from wmh2017.config.training_config import InputModality, resolve_input_modalities

REPO_ROOT = Path(__file__).resolve().parents[2]
A2CV_CONFIG = REPO_ROOT / "configs" / "experiments" / "cv" / "exp_a2cv_cosine_fold0.yaml"
RC2_CONFIG = REPO_ROOT / "configs" / "experiments" / "recall" / "exp_rc2_cosine_fold0.yaml"


def _resolve_from_yaml(path: Path) -> tuple[InputModality, ...]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_input_modalities(cfg["data"])


def test_a2cv_config_resolves_to_single_flair_modality():
    modalities = _resolve_from_yaml(A2CV_CONFIG)
    assert modalities == (InputModality(name="image", manifest_key="flair_pre_path", required=True),)


def test_rc2_config_resolves_to_single_flair_modality():
    modalities = _resolve_from_yaml(RC2_CONFIG)
    assert modalities == (InputModality(name="image", manifest_key="flair_pre_path", required=True),)


def test_legacy_default_image_key():
    assert resolve_input_modalities({}) == (InputModality(name="image", manifest_key="flair_pre_path", required=True),)


def test_explicit_input_modalities_parsed_in_order():
    data_cfg = {
        "input_modalities": [
            {"name": "flair", "manifest_key": "flair_pre_path"},
            {"name": "t1", "manifest_key": "t1_pre_path", "required": False},
        ]
    }
    assert resolve_input_modalities(data_cfg) == (
        InputModality(name="flair", manifest_key="flair_pre_path", required=True),
        InputModality(name="t1", manifest_key="t1_pre_path", required=False),
    )
