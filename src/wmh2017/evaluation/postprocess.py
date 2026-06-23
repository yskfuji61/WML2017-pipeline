"""Binary post-processing for WMH2017 predictions (threshold + CC filter + rescue).

Extracted from ``cross_arch_ensemble`` so threshold sweeps and ensembles share one
post-processing implementation. The numeric behavior of :func:`post_process_binary`
is unchanged from the prior in-place definition.

Post-processing is metric-overfit prone (component removal can delete small lesions),
so it is validation-only and the chosen settings must be recorded with their split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label


@dataclass(frozen=True)
class PostProcessConfig:
    """Declarative post-processing settings (recorded alongside metrics)."""

    threshold: float
    min_component_size: int = 0
    adaptive_low_threshold: float = 0.0
    adaptive_high_volume_voxels: int = 0


def post_process_binary(
    prob: np.ndarray,
    *,
    threshold: float,
    min_size: int = 0,
    adaptive_low_thr: float = 0.0,
    adaptive_high_vol: int = 0,
) -> np.ndarray:
    """Threshold + optional CC filter + optional adaptive low-threshold rescue."""
    binary = (prob >= float(threshold)).astype(np.uint8)
    if adaptive_low_thr > 0 and adaptive_high_vol > 0 and int(binary.sum()) > adaptive_high_vol:
        binary = (prob >= float(adaptive_low_thr)).astype(np.uint8)
    if min_size > 0 and binary.sum() > 0:
        lbl, n = cc_label(binary)
        keep = np.zeros_like(binary)
        for i in range(1, n + 1):
            comp = lbl == i
            if comp.sum() >= min_size:
                keep[comp] = 1
        binary = keep
    return binary.astype(np.uint8)


def apply_post_process(prob: np.ndarray, config: PostProcessConfig) -> np.ndarray:
    """Apply :func:`post_process_binary` using a :class:`PostProcessConfig`."""
    return post_process_binary(
        prob,
        threshold=config.threshold,
        min_size=config.min_component_size,
        adaptive_low_thr=config.adaptive_low_threshold,
        adaptive_high_vol=config.adaptive_high_volume_voxels,
    )
