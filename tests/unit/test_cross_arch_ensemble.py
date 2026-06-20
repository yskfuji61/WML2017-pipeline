from __future__ import annotations

import numpy as np
import pandas as pd

from wmh2017.evaluation.cross_arch_ensemble import evaluate_fused_predictions, fuse_probability_maps


def test_fuse_probability_maps_weighted_average():
    a = np.array([0.0, 0.8], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    fused = fuse_probability_maps(a, b, secondary_weight=0.5)
    np.testing.assert_allclose(fused, np.array([0.5, 0.4], dtype=np.float32))


def test_evaluate_fused_predictions_on_synthetic_case(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "flair_pre_path": str(tmp_path / "img.npy"),
                "wmh_path": str(tmp_path / "lbl.npy"),
            }
        ]
    )
    split = pd.DataFrame([{"case_id": "case001", "assigned_split": "val"}])
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    img = np.zeros((4, 8, 8), dtype=np.float32)
    lbl = np.zeros((4, 8, 8), dtype=np.uint8)
    lbl[2, 2:6, 2:6] = 1
    np.save(tmp_path / "img.npy", img)
    np.save(tmp_path / "lbl.npy", lbl)

    primary = np.zeros((4, 8, 8), dtype=np.float32)
    secondary = np.zeros((4, 8, 8), dtype=np.float32)
    primary[2, 2:6, 2:6] = 0.9
    secondary[2, 2:6, 2:6] = 0.8
    pdir = tmp_path / "primary"
    sdir = tmp_path / "secondary"
    pdir.mkdir()
    sdir.mkdir()
    np.savez_compressed(pdir / "case001.npz", probs=primary)
    np.savez_compressed(sdir / "case001.npz", probs=secondary)

    summary = evaluate_fused_predictions(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        primary_probs_dir=pdir,
        secondary_probs_dir=sdir,
        threshold=0.5,
        secondary_weight=0.5,
    )
    assert summary["mean_dice"] > 0.9
