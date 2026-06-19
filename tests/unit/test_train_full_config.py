"""Config validation for full training mode (no torch import required)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(rel_path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def test_smoke_config_defaults_to_smoke_mode() -> None:
    cfg = _load_yaml("configs/wmh2017_monai_smoke.yaml")
    assert cfg["data"]["split_manifest"] == "data/splits/wmh2017_train_val_seed42.csv"
    assert cfg["training"].get("mode", "smoke") == "smoke" or "mode" not in cfg["training"]


def test_tiny_smoke_config_uses_seed42_split() -> None:
    cfg = _load_yaml("configs/wmh2017_monai_tiny_smoke.yaml")
    assert cfg["run"]["seed"] == 42
    assert cfg["data"]["split_manifest"] == "data/splits/wmh2017_train_val_seed42.csv"


def test_full_config_has_required_keys() -> None:
    cfg = _load_yaml("configs/wmh2017_monai_unet3d_full.yaml")
    training = cfg["training"]
    assert training["mode"] == "full"
    assert training["max_epochs"] >= 1
    assert "early_stopping_patience" in training
    assert cfg["data"]["cache_rate"] >= 0.0
    assert cfg["data"]["num_workers"] >= 0
    assert "use_amp" in training
