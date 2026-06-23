"""Typed input-modality boundary for WMH2017 configs.

Existing configs declare a single FLAIR channel via ``data.image_key``. This module
normalizes both the legacy single-key form and an explicit ``data.input_modalities``
list into one internal type so downstream code (case records, model factory,
inference input builder, transforms) never branches on raw config dicts.

Backward compatibility is exact: a config without ``input_modalities`` resolves to a
single modality named ``"image"`` whose ``manifest_key`` is ``data.image_key`` (default
``flair_pre_path``), reproducing the prior FLAIR-only behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_IMAGE_MANIFEST_KEY = "flair_pre_path"
LEGACY_IMAGE_NAME = "image"


@dataclass(frozen=True)
class InputModality:
    """One model input channel resolved from the dataset manifest.

    ``name`` is the MONAI dictionary key used for this channel during transforms
    (the legacy single-channel name is ``"image"``). ``manifest_key`` is the column
    in the dataset manifest CSV that holds the path for this modality.
    """

    name: str
    manifest_key: str
    required: bool = True


def resolve_input_modalities(data_cfg: Mapping[str, Any]) -> tuple[InputModality, ...]:
    """Resolve the input modalities for a ``data`` config block.

    If ``input_modalities`` is present it is used verbatim. Otherwise the legacy
    single-key path is taken, yielding ``(InputModality("image", image_key),)``.
    """
    raw = data_cfg.get("input_modalities")
    if raw:
        return tuple(
            InputModality(
                name=str(item["name"]),
                manifest_key=str(item["manifest_key"]),
                required=bool(item.get("required", True)),
            )
            for item in raw
        )
    image_key = str(data_cfg.get("image_key", DEFAULT_IMAGE_MANIFEST_KEY))
    return (InputModality(name=LEGACY_IMAGE_NAME, manifest_key=image_key, required=True),)


def modality_keys(input_modalities: tuple[InputModality, ...]) -> tuple[str, ...]:
    """Return the ordered MONAI dictionary keys for the given modalities."""
    return tuple(m.name for m in input_modalities)


def modalities_to_payload(input_modalities: tuple[InputModality, ...]) -> list[dict[str, Any]]:
    """Serialize modalities for checkpoint/evidence payloads (JSON/torch-safe)."""
    return [{"name": m.name, "manifest_key": m.manifest_key, "required": m.required} for m in input_modalities]
