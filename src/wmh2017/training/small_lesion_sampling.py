"""Default-off small-lesion-aware positive-crop sampling support.

Biases *positive* crop centers toward small lesion components by precomputing oversampled
``fg_indices`` for MONAI's ``RandCropByPosNegLabeld`` (which selects positive centers uniformly
from ``fg_indices``). All cropping / padding / neg-sampling / num_samples behavior stays MONAI's.
When disabled (``small_lesion_center_prob`` 0/absent) this module is not used and the existing
pipeline is unchanged. Sampling-only: loss/model/LR/threshold/postprocess are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wmh2017.evaluation.lesion_metrics import connected_components


def small_lesion_flat_indices(label_fg_3d: np.ndarray, max_voxels: int, *, connectivity: int = 26) -> np.ndarray:
    """Flat (ravel) indices of foreground voxels in components with size <= ``max_voxels``."""
    fg = np.asarray(label_fg_3d) > 0
    if not fg.any():
        return np.empty(0, dtype=np.int64)
    cc, n = connected_components(fg.astype(np.uint8), connectivity=connectivity)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    sizes = np.bincount(cc.ravel())
    small_labels = {lbl for lbl in range(1, n + 1) if int(sizes[lbl]) <= int(max_voxels)}
    if not small_labels:
        return np.empty(0, dtype=np.int64)
    small_mask = np.isin(cc, list(small_labels))
    return np.flatnonzero(small_mask.ravel()).astype(np.int64)


def map_fg_bg_flat_indices(label_fg_3d: np.ndarray, brain_mask_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Foreground / background flat indices (MONAI map_binary_to_indices semantics).

    fg = voxels with label>0; bg = voxels with label==0 AND inside the brain mask.
    """
    fg = np.asarray(label_fg_3d) > 0
    brain = np.asarray(brain_mask_3d) > 0
    fg_idx = np.flatnonzero(fg.ravel()).astype(np.int64)
    bg_idx = np.flatnonzero((~fg & brain).ravel()).astype(np.int64)
    return fg_idx, bg_idx


def build_biased_fg_indices(fg_idx: np.ndarray, small_idx: np.ndarray, small_center_prob: float) -> np.ndarray:
    """Return fg indices oversampled so P(positive center in small lesion) ~= small_center_prob.

    - prob <= 0 or no small lesions  -> ``fg_idx`` unchanged (fallback).
    - prob >= 1                      -> ``small_idx`` only (positives always centered on small).
    - otherwise oversample small voxels among the fg index multiset to the target ratio.
    """
    fg_idx = np.asarray(fg_idx, dtype=np.int64)
    small_idx = np.asarray(small_idx, dtype=np.int64)
    p = float(small_center_prob)
    if p <= 0.0 or small_idx.size == 0:
        return fg_idx
    if p >= 1.0:
        return small_idx
    non_small = np.setdiff1d(fg_idx, small_idx, assume_unique=False)
    n_small = int(small_idx.size)
    n_non = int(non_small.size)
    if n_non == 0:
        return fg_idx  # all foreground is already small
    # want reps*S / (reps*S + L) = p  ->  reps = p*L / ((1-p)*S)
    reps = max(1, int(round(p * n_non / ((1.0 - p) * n_small))))
    return np.concatenate([non_small, np.tile(small_idx, reps)]).astype(np.int64)


def resolve_small_lesion_sampling_cfg(sampling_cfg: dict[str, Any]) -> tuple[bool, float, int]:
    """Return (enabled, small_center_prob, max_voxels); raise on invalid config. Default-off."""
    cfg = sampling_cfg if isinstance(sampling_cfg, dict) else {}
    prob = float(cfg.get("small_lesion_center_prob", 0.0))
    max_voxels = int(cfg.get("small_lesion_max_voxels", 10))
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"small_lesion_center_prob must be in [0, 1]; got {prob}")
    if max_voxels < 1:
        raise ValueError(f"small_lesion_max_voxels must be >= 1; got {max_voxels}")
    return (prob > 0.0, prob, max_voxels)


class SmallLesionFgBgIndicesd:
    """Dictionary transform: write biased ``fg_indices`` + ``bg_indices`` for the crop.

    Reads ``label`` (channel-first foreground 0/1) and ``image`` (channel-first) to derive the
    brain mask; writes oversampled fg indices (small-lesion-biased) and standard bg indices for
    ``RandCropByPosNegLabeld(fg_indices_key=..., bg_indices_key=...)``. Other keys are untouched.
    """

    def __init__(
        self,
        *,
        label_key: str = "label",
        image_key: str = "image",
        max_voxels: int = 10,
        small_center_prob: float = 0.0,
        image_threshold: float = 0.0,
        connectivity: int = 26,
        fg_indices_key: str = "fg_indices",
        bg_indices_key: str = "bg_indices",
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.max_voxels = int(max_voxels)
        self.small_center_prob = float(small_center_prob)
        self.image_threshold = float(image_threshold)
        self.connectivity = int(connectivity)
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        label = np.asarray(out[self.label_key])
        label3d = label[0] if label.ndim == 4 else label
        image = np.asarray(out[self.image_key])
        # brain = any channel above threshold
        if image.ndim == 4:
            brain = (image > self.image_threshold).any(axis=0)
        else:
            brain = image > self.image_threshold

        fg_idx, bg_idx = map_fg_bg_flat_indices(label3d > 0, brain)
        small_idx = small_lesion_flat_indices(label3d > 0, self.max_voxels, connectivity=self.connectivity)
        biased_fg = build_biased_fg_indices(fg_idx, small_idx, self.small_center_prob)
        out[self.fg_indices_key] = biased_fg.astype(np.int64)
        out[self.bg_indices_key] = bg_idx.astype(np.int64)
        return out
