from __future__ import annotations

import numpy as np
import pandas as pd

from wmh2017.evaluation.threshold_sweep import (
    default_threshold_grid,
    select_best_threshold,
    sweep_thresholds,
    write_threshold_sweep_artifacts,
)
from wmh2017.inference.export_probabilities import save_case_probability_map


def test_default_threshold_grid():
    grid = default_threshold_grid()
    assert grid[0] == 0.05
    assert grid[-1] == 0.5
    assert len(grid) == 10


def test_threshold_sweep_selects_lower_threshold_for_under_segmentation(tmp_path):
    probs = np.zeros((8, 8, 8), dtype=np.float32)
    probs[2:6, 2:6, 2:6] = 0.35

    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 1

    label_path = tmp_path / "case001_wmh.npy"
    np.save(label_path, label)

    probs_dir = tmp_path / "probs"
    probs_dir.mkdir()
    save_case_probability_map(probs, probs_dir / "case001.npz")

    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "wmh_path": str(label_path),
            }
        ]
    )
    split = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "assigned_split": "val",
            }
        ]
    )
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    summary_df, _per_case_df = sweep_thresholds(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        assigned_split="val",
        thresholds=[0.5, 0.35, 0.2],
    )
    best = select_best_threshold(summary_df)
    assert best["threshold"] == 0.2
    assert best["mean_dice"] > 0.99

    out_dir = tmp_path / "sweep"
    payload = write_threshold_sweep_artifacts(
        out_dir=out_dir,
        summary_df=summary_df,
        per_case_df=_per_case_df,
        best=best,
        run_id="test_run",
        probs_dir=probs_dir,
        training_threshold=0.5,
    )
    assert payload["threshold_policy"]["sweep_best_threshold"] == 0.2
    assert (out_dir / "threshold_sweep_best.json").exists()


def test_threshold_sweep_rejects_test_split(tmp_path):
    label_path = tmp_path / "case001_wmh.npy"
    np.save(label_path, np.zeros((4, 4, 4), dtype=np.uint8))
    probs_dir = tmp_path / "probs"
    probs_dir.mkdir()
    save_case_probability_map(np.zeros((4, 4, 4), dtype=np.float32), probs_dir / "case001.npz")

    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "test",
                "wmh_path": str(label_path),
            }
        ]
    )
    split = pd.DataFrame([{"case_id": "case001", "assigned_split": "val"}])
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    split.to_csv(tmp_path / "split.csv", index=False)

    try:
        sweep_thresholds(
            manifest_csv=tmp_path / "manifest.csv",
            split_csv=tmp_path / "split.csv",
            probs_dir=probs_dir,
        )
        raise AssertionError("expected ValueError for test split")
    except ValueError as exc:
        assert "challenge_split=test" in str(exc)
