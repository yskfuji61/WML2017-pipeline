"""Shared input-tensor construction for validation and probability export.

Training-time validation inference and standalone probability export must build the
model input identically. This module is the single place that loads, normalizes, and
stacks modalities into a channel-first volume, so a future multi-channel change (e.g.
FLAIR+T1) lands in one place instead of being duplicated.

Single-channel parity: with one key, :func:`load_normalized_input_volume` returns a
``(1, Z, Y, X)`` volume and :func:`to_batched_tensor` makes it ``(1, 1, Z, Y, X)`` —
exactly the prior ``x[None, None]`` shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from wmh2017.data.preprocessing import normalize_nonzero_channelwise
from wmh2017.io.images import load_array


def load_normalized_input_volume(
    *,
    image_paths: Mapping[str, str],
    input_keys: tuple[str, ...],
) -> np.ndarray:
    """Load and per-channel normalize the given modalities into a (C, Z, Y, X) volume."""
    if not input_keys:
        raise ValueError("input_keys must contain at least one modality key")
    channels: list[np.ndarray] = []
    for key in input_keys:
        if key not in image_paths:
            raise KeyError(f"missing image path for input key '{key}'")
        arr = load_array(image_paths[key])
        channels.append(normalize_nonzero_channelwise(arr)[None])
    return np.concatenate(channels, axis=0).astype(np.float32)


def to_batched_tensor(torch: Any, volume_czyx: np.ndarray, device: Any) -> Any:
    """Add a batch dim and move to device: (C, Z, Y, X) -> (1, C, Z, Y, X) tensor."""
    return torch.from_numpy(volume_czyx[None].astype(np.float32)).to(device)
