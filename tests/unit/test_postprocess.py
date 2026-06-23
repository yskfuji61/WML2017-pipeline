from __future__ import annotations

import numpy as np
import pytest

from wmh2017.evaluation.postprocess import PostProcessConfig, apply_post_process, post_process_binary


@pytest.mark.requires_scipy
def test_threshold_only_keeps_above_threshold():
    prob = np.array([[[0.2, 0.6], [0.9, 0.1]]], dtype=np.float32)
    out = post_process_binary(prob, threshold=0.5)
    np.testing.assert_array_equal(out, np.array([[[0, 1], [1, 0]]], dtype=np.uint8))


@pytest.mark.requires_scipy
def test_min_component_size_removes_small_component():
    prob = np.zeros((1, 6, 6), dtype=np.float32)
    # One isolated voxel (size 1) and a 2x2 block (size 4).
    prob[0, 0, 0] = 0.9
    prob[0, 3:5, 3:5] = 0.9
    out = post_process_binary(prob, threshold=0.5, min_size=2)
    assert out[0, 0, 0] == 0  # small component dropped
    assert out[0, 3:5, 3:5].sum() == 4  # large component kept


@pytest.mark.requires_scipy
def test_adaptive_low_threshold_rescue_triggers_above_volume_cap():
    prob = np.full((1, 4, 4), 0.4, dtype=np.float32)
    prob[0, :3, :3] = 0.9  # 9 voxels above 0.5
    # At threshold 0.5, 9 voxels survive (>cap=5) so the low-threshold rescue fires,
    # re-thresholding at 0.3 which captures all 16 voxels (background is 0.4).
    rescued = post_process_binary(prob, threshold=0.5, adaptive_low_thr=0.3, adaptive_high_vol=5)
    assert rescued.sum() == 16


@pytest.mark.requires_scipy
def test_adaptive_rescue_does_not_fire_below_volume_cap():
    prob = np.full((1, 4, 4), 0.1, dtype=np.float32)
    prob[0, 0, 0] = 0.9  # only 1 voxel above 0.5, below cap
    out = post_process_binary(prob, threshold=0.5, adaptive_low_thr=0.3, adaptive_high_vol=5)
    assert out.sum() == 1


@pytest.mark.requires_scipy
def test_config_wrapper_matches_function():
    prob = np.zeros((1, 6, 6), dtype=np.float32)
    prob[0, 0, 0] = 0.9
    prob[0, 3:5, 3:5] = 0.9
    config = PostProcessConfig(threshold=0.5, min_component_size=2)
    np.testing.assert_array_equal(
        apply_post_process(prob, config),
        post_process_binary(prob, threshold=0.5, min_size=2),
    )
