"""Lesion-wise metric utilities for WMH2017 local validation.

Claim boundary:
- Individual lesions are 3D connected components in the binary WMH mask.
- The official challenge metric implementation must be referenced before SOTA
  or leaderboard-comparable claims. This module is intended for local sanity and
  reproducible validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wmh2017.data.label_policy import wmh_foreground_mask

# (name, min_voxels inclusive, max_voxels exclusive); max_voxels=None means unbounded.
DEFAULT_SIZE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("small", 1, 10),
    ("medium", 10, 50),
    ("large", 50, None),
)


@dataclass(frozen=True)
class LesionMetricPolicy:
    foreground_label: int = 1
    ignore_label: int = 2
    connectivity: int = 26
    matching_rule: str = "overlap_gt_0"


def _structure_for_connectivity(ndim: int, connectivity: int) -> np.ndarray:
    from scipy.ndimage import generate_binary_structure

    if ndim != 3:
        return generate_binary_structure(ndim, ndim)
    if connectivity <= 6:
        return generate_binary_structure(3, 1)
    if connectivity <= 18:
        return generate_binary_structure(3, 2)
    return generate_binary_structure(3, 3)


def connected_components(mask: np.ndarray, connectivity: int = 26) -> tuple[np.ndarray, int]:
    """Return connected-component labels and count for a binary mask."""
    from scipy.ndimage import label as cc_label

    mask_b = np.asarray(mask).astype(bool)
    structure = _structure_for_connectivity(mask_b.ndim, connectivity)
    labeled, count = cc_label(mask_b, structure=structure)
    return labeled, int(count)


def lesion_recall_f1_binary(pred: np.ndarray, target: np.ndarray, *, connectivity: int = 26) -> dict[str, float | int]:
    """Compute lesion-wise recall and F1 using overlap-based matching.

    A target lesion is counted as detected if any predicted lesion overlaps it.
    A predicted lesion is counted as true positive if it overlaps any target lesion.
    This simple matching policy is explicit and testable; before official claims,
    compare it to the official WMH evaluation code.
    """
    pred_cc, n_pred = connected_components(pred, connectivity=connectivity)
    target_cc, n_target = connected_components(target, connectivity=connectivity)

    if n_target == 0 and n_pred == 0:
        return {"lesion_recall": 1.0, "lesion_f1": 1.0, "tp_target": 0, "tp_pred": 0, "n_target": 0, "n_pred": 0}
    if n_target == 0:
        return {"lesion_recall": 1.0, "lesion_f1": 0.0, "tp_target": 0, "tp_pred": 0, "n_target": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"lesion_recall": 0.0, "lesion_f1": 0.0, "tp_target": 0, "tp_pred": 0, "n_target": n_target, "n_pred": 0}

    target_detected = set(np.unique(target_cc[pred_cc > 0]).tolist()) - {0}
    pred_matched = set(np.unique(pred_cc[target_cc > 0]).tolist()) - {0}

    tp_target = len(target_detected)
    tp_pred = len(pred_matched)
    recall = tp_target / n_target if n_target else 1.0
    precision = tp_pred / n_pred if n_pred else 1.0
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    return {
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "tp_target": int(tp_target),
        "tp_pred": int(tp_pred),
        "n_target": int(n_target),
        "n_pred": int(n_pred),
    }


def lesion_recall_f1_wmh_label1(
    pred_mask: np.ndarray, target_mask: np.ndarray, *, connectivity: int = 26
) -> dict[str, float | int]:
    """Lesion recall/F1 for WMH label 1 only. label==2 is not foreground."""
    return lesion_recall_f1_binary(
        wmh_foreground_mask(pred_mask),
        wmh_foreground_mask(target_mask),
        connectivity=connectivity,
    )


def lesion_recall_by_size_bins(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    bins: Sequence[tuple[str, int, int | None]] = DEFAULT_SIZE_BINS,
    connectivity: int = 26,
) -> list[dict[str, float | int | str | None]]:
    """Recall of GT lesions grouped by ground-truth connected-component voxel size.

    A GT lesion is "detected" if any predicted lesion overlaps it (same overlap rule as
    :func:`lesion_recall_f1_binary`). Bins partition GT lesions by their voxel count, so a
    recall drop in the smallest bin surfaces small-lesion deletion (e.g. from component
    filtering). ``recall`` is ``None`` for an empty bin. Diagnostic, validation-only.
    """
    pred_cc, _ = connected_components(pred, connectivity=connectivity)
    target_cc, n_target = connected_components(target, connectivity=connectivity)

    detected = set(np.unique(target_cc[pred_cc > 0]).tolist()) - {0}
    # sizes[label] = voxel count of that GT component; index 0 is background.
    sizes = np.bincount(target_cc.ravel()) if n_target else np.array([0], dtype=np.int64)

    rows: list[dict[str, float | int | str | None]] = []
    for name, lo, hi in bins:
        labels_in_bin = [
            lbl for lbl in range(1, n_target + 1) if int(sizes[lbl]) >= lo and (hi is None or int(sizes[lbl]) < hi)
        ]
        n_t = len(labels_in_bin)
        n_d = sum(1 for lbl in labels_in_bin if lbl in detected)
        recall: float | None = (n_d / n_t) if n_t else None
        rows.append(
            {
                "bin": name,
                "min_voxels": int(lo),
                "max_voxels": (None if hi is None else int(hi)),
                "n_target": int(n_t),
                "n_detected": int(n_d),
                "recall": recall,
            }
        )
    return rows


def lesion_recall_by_size_bins_wmh_label1(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    bins: Sequence[tuple[str, int, int | None]] = DEFAULT_SIZE_BINS,
    connectivity: int = 26,
) -> list[dict[str, float | int | str | None]]:
    """Size-binned lesion recall for WMH label 1 only. label==2 is not foreground."""
    return lesion_recall_by_size_bins(
        wmh_foreground_mask(pred_mask),
        wmh_foreground_mask(target_mask),
        bins=bins,
        connectivity=connectivity,
    )


def not_implemented_message() -> str:
    return (
        "Lesion-wise metrics are implemented for local validation, but must be "
        "cross-checked with the official WMH evaluation code before SOTA claims."
    )
