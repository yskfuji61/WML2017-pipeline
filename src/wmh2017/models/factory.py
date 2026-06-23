"""MONAI UNet construction with input-channel validation.

Centralizes model building so ``model.in_channels`` is checked against the resolved
input modalities. This catches the classic multi-modality mistake (declaring two
modalities but leaving ``in_channels: 1``) with a fail-fast error.

Backward compatibility: ``build_unet(monai, cfg)`` with no modalities behaves exactly
like the prior ``train_monai._build_model`` (``in_channels`` defaults to 1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wmh2017.config.training_config import InputModality


def resolve_in_channels(
    *,
    model_cfg: Mapping[str, Any],
    input_modalities: tuple[InputModality, ...],
) -> int:
    """Return the model input-channel count, validating config vs modalities.

    If ``in_channels`` is configured it must equal the number of input modalities;
    otherwise the modality count is inferred.
    """
    inferred = len(input_modalities)
    configured = model_cfg.get("in_channels")
    if configured is None:
        return inferred
    configured_int = int(configured)
    if configured_int != inferred:
        raise ValueError(
            f"model.in_channels={configured_int} does not match the number of "
            f"input_modalities={inferred} ({[m.name for m in input_modalities]})"
        )
    return configured_int


def build_unet(
    monai: dict[str, Any],
    cfg: dict[str, Any],
    *,
    input_modalities: tuple[InputModality, ...] | None = None,
) -> Any:
    """Build a MONAI UNet from ``cfg.model``.

    When ``input_modalities`` is provided, ``in_channels`` is validated/inferred via
    :func:`resolve_in_channels`. When omitted (legacy callers), ``model.in_channels``
    is used directly with a default of 1 — identical to the prior behavior.
    """
    m = cfg.get("model", {})
    if input_modalities is None:
        in_channels = int(m.get("in_channels", 1))
    else:
        in_channels = resolve_in_channels(model_cfg=m, input_modalities=input_modalities)
    return monai["UNet"](
        spatial_dims=int(m.get("spatial_dims", 3)),
        in_channels=in_channels,
        out_channels=int(m.get("out_channels", 2)),
        channels=tuple(m.get("channels", [8, 16, 32])),
        strides=tuple(m.get("strides", [2, 2])),
        num_res_units=int(m.get("num_res_units", 1)),
    )


def assert_checkpoint_modality_compat(
    *,
    checkpoint_modalities: Sequence[Mapping[str, Any]] | None,
    config_modalities: tuple[InputModality, ...],
) -> None:
    """Fail fast if a checkpoint's channel count disagrees with the config.

    Prevents loading a 1-channel checkpoint under a 2-channel config (and vice versa)
    with a readable message instead of an opaque ``load_state_dict`` shape error.
    ``checkpoint_modalities`` is the list persisted in the checkpoint payload; ``None``
    (legacy checkpoints without metadata) is treated as a single channel.
    """
    config_count = len(config_modalities)
    checkpoint_count = 1 if checkpoint_modalities is None else len(checkpoint_modalities)
    if checkpoint_count != config_count:
        raise ValueError(
            f"checkpoint was trained with {checkpoint_count} input channel(s) but the "
            f"config resolves to {config_count} input_modalities "
            f"({[m.name for m in config_modalities]}); refusing to load mismatched weights"
        )
