"""Missing-prediction policy: full fails on missing, smoke may skip.

These tests exercise the underlying evaluation contract (evaluate_predictions) plus
the e2e stage's mode-driven decision to add --skip-missing-predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wmh2017.e2e.stages import _train_config_mode
from wmh2017.evaluation.evaluate_predictions import evaluate_predictions


def _make_two_case_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two val cases; only case001 has a prediction file."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    rows_manifest = []
    rows_split = []
    for cid in ("case001", "case002"):
        label = np.zeros((4, 4, 4), dtype=np.uint8)
        label[1, 1, 1] = 1
        label_path = tmp_path / f"{cid}_wmh.npy"
        np.save(label_path, label)
        rows_manifest.append(
            {"case_id": cid, "challenge_split": "training", "site": "fixture", "wmh_path": str(label_path)}
        )
        rows_split.append({"case_id": cid, "assigned_split": "val", "site": "fixture"})

    # Only case001 gets a prediction.
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    pred[1, 1, 1] = 1
    np.save(pred_dir / "case001_pred.npy", pred)

    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows_manifest).to_csv(manifest_csv, index=False)
    pd.DataFrame(rows_split).to_csv(split_csv, index=False)
    return manifest_csv, split_csv, pred_dir


def test_full_policy_fails_on_missing_prediction(tmp_path: Path) -> None:
    manifest_csv, split_csv, pred_dir = _make_two_case_fixture(tmp_path)
    # Full mode passes no skip flag -> evaluate must fail on the missing case.
    with pytest.raises(FileNotFoundError):
        evaluate_predictions(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            prediction_dir=pred_dir,
            out_dir=tmp_path / "eval",
            run_id="full_run",
            assigned_split="val",
            skip_missing_predictions=False,
        )


def test_smoke_policy_skips_and_records_coverage(tmp_path: Path) -> None:
    manifest_csv, split_csv, pred_dir = _make_two_case_fixture(tmp_path)
    out = evaluate_predictions(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        prediction_dir=pred_dir,
        out_dir=tmp_path / "eval",
        run_id="smoke_run",
        assigned_split="val",
        skip_missing_predictions=True,
    )
    coverage = out["prediction_coverage"]
    assert coverage["expected_cases"] == 2
    assert coverage["evaluated_cases"] == 1
    assert coverage["missing_predictions"] == ["case002"]
    assert coverage["full_coverage"] is False


def test_train_config_mode_drives_skip_decision(tmp_path: Path) -> None:
    full_cfg = tmp_path / "full.yaml"
    full_cfg.write_text("training:\n  mode: full\n", encoding="utf-8")
    smoke_cfg = tmp_path / "smoke.yaml"
    smoke_cfg.write_text("training:\n  max_epochs: 1\n", encoding="utf-8")

    assert _train_config_mode(full_cfg) == "full"
    assert _train_config_mode(smoke_cfg) == "smoke"
    assert _train_config_mode(None) == "smoke"
    assert _train_config_mode(tmp_path / "missing.yaml") == "smoke"
