from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wmh2017.evaluation.lesion_metrics import lesion_recall_by_size_bins
from wmh2017.evaluation.threshold_sweep import (
    lesion_size_bin_audit,
    select_best_threshold,
    sweep_thresholds,
    write_lesion_size_bin_audit_artifact,
    write_threshold_sweep_artifacts,
)
from wmh2017.inference.export_probabilities import save_case_probability_map


def _small_and_large_target() -> np.ndarray:
    target = np.zeros((4, 12, 12), dtype=np.uint8)
    target[:, 0:4, 0:4] = 1  # large lesion: 4*4*4 = 64 voxels
    target[0, 10:12, 10:12] = 1  # small lesion: 4 voxels, separated
    return target


# ---- pure metric ----


def test_size_bins_recall_detects_only_large():
    target = _small_and_large_target()
    pred = np.zeros_like(target)
    pred[:, 0:4, 0:4] = 1  # detect only the large lesion
    rows = {r["bin"]: r for r in lesion_recall_by_size_bins(pred, target)}
    assert rows["small"]["n_target"] == 1 and rows["small"]["n_detected"] == 0
    assert rows["small"]["recall"] == 0.0
    assert rows["large"]["n_target"] == 1 and rows["large"]["recall"] == 1.0
    assert rows["medium"]["n_target"] == 0 and rows["medium"]["recall"] is None


def test_size_bins_full_detection():
    target = _small_and_large_target()
    rows = {r["bin"]: r for r in lesion_recall_by_size_bins(target.copy(), target)}
    assert rows["small"]["recall"] == 1.0
    assert rows["large"]["recall"] == 1.0


# ---- audit over a split ----


def _write_case(tmp_path: Path, *, challenge_split: str = "training") -> tuple[Path, Path, Path]:
    target = _small_and_large_target()
    probs = np.zeros((4, 12, 12), dtype=np.float32)
    probs[:, 0:4, 0:4] = 0.9  # large predicted
    probs[0, 10:12, 10:12] = 0.9  # small predicted (4 voxels)

    label_path = tmp_path / "case001_wmh.npy"
    np.save(label_path, target)
    probs_dir = tmp_path / "probs"
    probs_dir.mkdir(exist_ok=True)
    save_case_probability_map(probs, probs_dir / "case001.npz")

    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame([{"case_id": "case001", "challenge_split": challenge_split, "wmh_path": str(label_path)}]).to_csv(
        manifest_csv, index=False
    )
    pd.DataFrame([{"case_id": "case001", "assigned_split": "val"}]).to_csv(split_csv, index=False)
    return manifest_csv, split_csv, probs_dir


def test_audit_flags_small_lesion_regression(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path)
    audit = lesion_size_bin_audit(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        threshold=0.5,
        min_component_size=10,  # deletes the 4-voxel small lesion
    )
    by_bin = {r["bin"]: r for r in audit["per_bin"]}
    assert by_bin["small"]["baseline_recall"] == 1.0
    assert by_bin["small"]["post_recall"] == 0.0
    assert by_bin["small"]["delta_recall"] == -1.0
    assert by_bin["large"]["post_recall"] == 1.0
    assert audit["small_lesion_recall_regressed"] is True


def test_audit_no_regression_when_filter_off(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path)
    audit = lesion_size_bin_audit(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        threshold=0.5,
        min_component_size=0,
    )
    by_bin = {r["bin"]: r for r in audit["per_bin"]}
    assert by_bin["small"]["baseline_recall"] == by_bin["small"]["post_recall"] == 1.0
    assert audit["small_lesion_recall_regressed"] is False


def test_audit_rejects_test_split(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path, challenge_split="test")
    with pytest.raises(ValueError, match="must not be used"):
        lesion_size_bin_audit(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            probs_dir=probs_dir,
            threshold=0.5,
            min_component_size=10,
        )


def test_audit_artifact_hash_stable(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path)
    audit = lesion_size_bin_audit(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        threshold=0.5,
        min_component_size=10,
    )
    p1 = write_lesion_size_bin_audit_artifact(out_dir=tmp_path / "a", audit=audit, run_id="r1")
    p2 = write_lesion_size_bin_audit_artifact(out_dir=tmp_path / "b", audit=audit, run_id="r1")
    assert (tmp_path / "a" / "lesion_size_bin_audit.json").exists()
    assert p1["artifact_hash"] == p2["artifact_hash"]
    assert p1["threshold_best_is_checkpoint_best"] is False


# ---- sweep postprocess knob ----


def test_sweep_min_component_size_drops_small_lesion_recall(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path)
    base, _ = sweep_thresholds(manifest_csv=manifest_csv, split_csv=split_csv, probs_dir=probs_dir, thresholds=[0.5])
    filtered, _ = sweep_thresholds(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        thresholds=[0.5],
        min_component_size=10,
    )
    base_recall = float(base.iloc[0]["mean_lesion_recall"])
    filtered_recall = float(filtered.iloc[0]["mean_lesion_recall"])
    assert base_recall == 1.0  # both lesions detected without filtering
    assert filtered_recall < base_recall  # small lesion deleted by the filter


def test_threshold_artifact_records_post_process_only_when_used(tmp_path: Path):
    manifest_csv, split_csv, probs_dir = _write_case(tmp_path)
    summary_df, per_case_df = sweep_thresholds(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        probs_dir=probs_dir,
        thresholds=[0.5],
        min_component_size=10,
    )
    best = select_best_threshold(summary_df)
    with_pp = write_threshold_sweep_artifacts(
        out_dir=tmp_path / "pp",
        summary_df=summary_df,
        per_case_df=per_case_df,
        best=best,
        run_id="r1",
        probs_dir=probs_dir,
        training_threshold=0.5,
        min_component_size=10,
    )
    without_pp = write_threshold_sweep_artifacts(
        out_dir=tmp_path / "nopp",
        summary_df=summary_df,
        per_case_df=per_case_df,
        best=best,
        run_id="r1",
        probs_dir=probs_dir,
        training_threshold=0.5,
    )
    assert with_pp["post_process"] == {"min_component_size": 10}
    assert "post_process" not in without_pp
