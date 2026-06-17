"""Metric utilities for WMH2017 validation and reproduction.

Claim boundary:
- These functions are suitable for local validation only until matched against the
  official WMH evaluation code and recorded in `metric_register_wmh2017.csv`.
- Foreground is strictly `label == 1`; `label == 2` is ignored and must never be
  converted to foreground via `mask > 0`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from wmh2017.data.label_policy import wmh_foreground_mask


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8, empty_score: float = 1.0) -> float:
    """Compute Dice for binary arrays.

    If both arrays are empty, return empty_score. This behavior must be reported
    in metric manifests when used for claims.
    """
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    pred_sum = int(pred_b.sum())
    target_sum = int(target_b.sum())
    if pred_sum == 0 and target_sum == 0:
        return float(empty_score)
    inter = int(np.logical_and(pred_b, target_b).sum())
    return float((2.0 * inter + eps) / (pred_sum + target_sum + eps))


def dice_wmh_label1(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Dice for WMH label 1 only. label==2 is not foreground."""
    return dice_binary(wmh_foreground_mask(pred_mask), wmh_foreground_mask(target_mask), eps=eps)


def absolute_volume_difference_percent(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    empty_score: float = 0.0,
) -> float:
    """Absolute volume difference in percent.

    The WMH challenge reports AVD (%). For a non-empty target:
        abs(V_pred - V_target) / V_target * 100.

    Spacing is accepted for explicitness. With identical spacing for pred/target,
    voxel counts and physical volumes produce the same percentage.
    """
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    voxel_volume = float(np.prod(spacing)) if spacing is not None else 1.0
    pred_volume = float(pred_b.sum()) * voxel_volume
    target_volume = float(target_b.sum()) * voxel_volume
    if target_volume == 0.0:
        return float(empty_score if pred_volume == 0.0 else np.inf)
    return float(abs(pred_volume - target_volume) / target_volume * 100.0)


def avd_wmh_label1(pred_mask: np.ndarray, target_mask: np.ndarray, *, spacing: tuple[float, ...] | None = None) -> float:
    """AVD (%) for WMH label 1 only."""
    return absolute_volume_difference_percent(
        wmh_foreground_mask(pred_mask),
        wmh_foreground_mask(target_mask),
        spacing=spacing,
    )


def _surface(mask: np.ndarray) -> np.ndarray:
    mask_b = np.asarray(mask).astype(bool)
    if not mask_b.any():
        return mask_b
    structure = np.ones((3,) * mask_b.ndim, dtype=bool)
    eroded = binary_erosion(mask_b, structure=structure, border_value=0)
    return np.logical_and(mask_b, np.logical_not(eroded))


def hausdorff95_binary(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    empty_score: float = 0.0,
    missing_score: float = np.inf,
) -> float:
    """Symmetric 95th percentile Hausdorff distance for binary masks.

    This is a local implementation for sanity checks. It must be compared with
    the official WMH evaluation implementation before benchmark claims.
    """
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)

    if not pred_b.any() and not target_b.any():
        return float(empty_score)
    if not pred_b.any() or not target_b.any():
        return float(missing_score)

    pred_surface = _surface(pred_b)
    target_surface = _surface(target_b)

    # Distance transform of the inverse surface gives distance to nearest surface voxel.
    sampling = spacing if spacing is not None else None
    dt_to_target = distance_transform_edt(~target_surface, sampling=sampling)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=sampling)

    distances = np.concatenate([
        dt_to_target[pred_surface].astype(float),
        dt_to_pred[target_surface].astype(float),
    ])
    if distances.size == 0:
        return float(missing_score)
    return float(np.percentile(distances, 95))


def hd95_wmh_label1(pred_mask: np.ndarray, target_mask: np.ndarray, *, spacing: tuple[float, ...] | None = None) -> float:
    """HD95 for WMH label 1 only."""
    return hausdorff95_binary(
        wmh_foreground_mask(pred_mask),
        wmh_foreground_mask(target_mask),
        spacing=spacing,
    )
