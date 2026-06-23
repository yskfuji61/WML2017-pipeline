from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wmh2017.evaluation.ensemble import (
    EnsembleMember,
    EnsembleSpec,
    evaluate_ensemble,
    fuse_probability_maps_weighted,
    spec_from_config,
    write_ensemble_evaluation_artifact,
)
from wmh2017.inference.export_probabilities import save_case_probability_map

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "ensemble" / "ensemble_a2cv_rc2_example.yaml"


def _members(weights: list[float]) -> tuple[EnsembleMember, ...]:
    return tuple(EnsembleMember(name=f"m{i}", probs_dir=f"/d{i}", weight=w) for i, w in enumerate(weights))


# ---- spec validation ----


def test_requires_two_members():
    with pytest.raises(ValueError, match="at least 2 members"):
        EnsembleSpec(members=_members([1.0]))


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        EnsembleSpec(members=_members([0.5, 0.4]))


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        EnsembleSpec(members=_members([1.2, -0.2]))


def test_duplicate_names_rejected():
    members = (
        EnsembleMember(name="dup", probs_dir="/a", weight=0.5),
        EnsembleMember(name="dup", probs_dir="/b", weight=0.5),
    )
    with pytest.raises(ValueError, match="unique"):
        EnsembleSpec(members=members)


def test_threshold_out_of_range_rejected():
    with pytest.raises(ValueError, match="threshold"):
        EnsembleSpec(members=_members([0.5, 0.5]), threshold=1.5)


# ---- fusion ----


def test_fuse_equal_weights_averages():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    fused = fuse_probability_maps_weighted([a, b], [0.5, 0.5])
    np.testing.assert_allclose(fused, np.full((2, 2), 0.5, dtype=np.float32))


def test_fuse_three_members():
    maps = [np.full((2,), v, dtype=np.float32) for v in (0.0, 0.6, 0.9)]
    fused = fuse_probability_maps_weighted(maps, [0.2, 0.3, 0.5])
    np.testing.assert_allclose(fused, np.full((2,), 0.2 * 0.0 + 0.3 * 0.6 + 0.5 * 0.9, dtype=np.float32))


def test_fuse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fuse_probability_maps_weighted([np.zeros((2, 2)), np.zeros((2, 3))], [0.5, 0.5])


def test_fuse_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        fuse_probability_maps_weighted([np.zeros((2,)), np.zeros((2,))], [1.0])


# ---- evaluation over a split ----


def _write_two_member_case(tmp_path: Path, *, challenge_split: str = "training") -> dict:
    label = np.zeros((4, 8, 8), dtype=np.uint8)
    label[1:3, 2:6, 2:6] = 1
    label_path = tmp_path / "case001_wmh.npy"
    np.save(label_path, label)

    probs = np.zeros((4, 8, 8), dtype=np.float32)
    probs[1:3, 2:6, 2:6] = 0.9
    dirs = {}
    for name in ("a", "b"):
        d = tmp_path / f"probs_{name}"
        d.mkdir()
        save_case_probability_map(probs, d / "case001.npz")
        dirs[name] = d

    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame([{"case_id": "case001", "challenge_split": challenge_split, "wmh_path": str(label_path)}]).to_csv(
        manifest_csv, index=False
    )
    pd.DataFrame([{"case_id": "case001", "assigned_split": "val"}]).to_csv(split_csv, index=False)
    return {"manifest": manifest_csv, "split": split_csv, "dirs": dirs}


def _spec_for(dirs: dict) -> EnsembleSpec:
    return EnsembleSpec(
        members=(
            EnsembleMember(name="a", probs_dir=str(dirs["a"]), weight=0.5),
            EnsembleMember(name="b", probs_dir=str(dirs["b"]), weight=0.5),
        ),
        threshold=0.5,
    )


def test_evaluate_ensemble_high_dice(tmp_path: Path):
    ctx = _write_two_member_case(tmp_path)
    summary = evaluate_ensemble(spec=_spec_for(ctx["dirs"]), manifest_csv=ctx["manifest"], split_csv=ctx["split"])
    assert summary["n_cases"] == 1
    assert summary["mean_dice"] > 0.99
    assert summary["weights"] == [0.5, 0.5]


def test_evaluate_ensemble_rejects_test_split(tmp_path: Path):
    ctx = _write_two_member_case(tmp_path, challenge_split="test")
    with pytest.raises(ValueError, match="must not be used"):
        evaluate_ensemble(spec=_spec_for(ctx["dirs"]), manifest_csv=ctx["manifest"], split_csv=ctx["split"])


def test_evaluate_ensemble_missing_member_map_raises(tmp_path: Path):
    ctx = _write_two_member_case(tmp_path)
    spec = EnsembleSpec(
        members=(
            EnsembleMember(name="a", probs_dir=str(ctx["dirs"]["a"]), weight=0.5),
            EnsembleMember(name="missing", probs_dir=str(tmp_path / "nope"), weight=0.5),
        ),
        threshold=0.5,
    )
    with pytest.raises(FileNotFoundError, match="missing prob map"):
        evaluate_ensemble(spec=spec, manifest_csv=ctx["manifest"], split_csv=ctx["split"])


# ---- config + artifact ----


def test_spec_from_real_example_config():
    import yaml

    spec = spec_from_config(yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8")))
    assert [m.name for m in spec.members] == ["a2cv", "rc2"]
    assert spec.weights() == [0.5, 0.5]
    assert spec.threshold == 0.5


def test_artifact_records_frozen_weights_and_hash(tmp_path: Path):
    ctx = _write_two_member_case(tmp_path)
    spec = _spec_for(ctx["dirs"])
    summary = evaluate_ensemble(spec=spec, manifest_csv=ctx["manifest"], split_csv=ctx["split"])
    payload = write_ensemble_evaluation_artifact(out_dir=tmp_path / "out", summary=summary, spec=spec, run_id="r1")
    assert (tmp_path / "out" / "ensemble_evaluation.json").exists()
    assert (tmp_path / "out" / "ensemble_eval_per_case.csv").exists()
    assert payload["weights_frozen"] is True
    assert "test weight tuning" in payload["prohibited_use"]
    assert "per_case" not in payload
    assert payload["artifact_hash"]
