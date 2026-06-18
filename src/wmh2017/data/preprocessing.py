"""Shared preprocessing policies for WMH2017 training and inference.

The same functions are used by MONAI dataset transforms and standalone
inference code to avoid silent train/inference drift.
"""
from __future__ import annotations

import numpy as np


def normalize_nonzero_channelwise(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-normalize non-zero voxels per channel.

    For WMH2017 FLAIR/T1 volumes the zero background is excluded from the
    statistics. This mirrors the intended MONAI `NormalizeIntensityd(nonzero=True,
    channel_wise=True)` behavior while keeping the policy testable without MONAI.

    Accepted shapes:
    - 3D: D/H/W single volume.
    - 4D: C/D/H/W channel-first volume.
    """
    x = np.asarray(image, dtype=np.float32).copy()
    if x.ndim == 4:
        for channel_idx in range(x.shape[0]):
            x[channel_idx] = _normalize_single_volume(x[channel_idx], eps=eps)
        return x
    return _normalize_single_volume(x, eps=eps)


def _normalize_single_volume(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    nonzero = volume[volume != 0]
    if nonzero.size:
        mean = float(nonzero.mean())
        std = max(float(nonzero.std()), eps)
    else:
        mean = float(volume.mean())
        std = max(float(volume.std()), eps)
    return ((volume - mean) / std).astype(np.float32, copy=False)
