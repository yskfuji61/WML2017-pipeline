from pathlib import Path

import numpy as np
import pandas as pd

from wmh2017.evaluation.evaluate_predictions import evaluate_predictions
from wmh2017.io.images import load_array, save_array_like


def test_evaluate_predictions_from_npy_fixture(tmp_path: Path):
    image = np.zeros((5, 5, 5), dtype=np.float32)
    label = np.zeros((5, 5, 5), dtype=np.uint8)
    pred = np.zeros((5, 5, 5), dtype=np.uint8)
    label[2, 2, 2] = 1
    pred[2, 2, 2] = 1

    image_path = tmp_path / "case001_flair.npy"
    label_path = tmp_path / "case001_wmh.npy"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred_path = pred_dir / "case001_pred.npy"

    np.save(image_path, image)
    np.save(label_path, label)
    np.save(pred_path, pred)

    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "site": "fixture",
                "scanner_code": "fixture",
                "flair_pre_path": str(image_path),
                "wmh_path": str(label_path),
            }
        ]
    )
    split = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "source_split": "training",
                "assigned_split": "val",
                "site": "fixture",
                "scanner_code": "fixture",
            }
        ]
    )
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    out = evaluate_predictions(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        prediction_dir=pred_dir,
        out_dir=tmp_path / "eval",
        run_id="fixture_run",
        assigned_split="val",
    )

    assert out["n_cases"] == 1
    assert out["mean_dice"] == 1.0
    assert Path(out["case_metrics_csv"]).exists()
    assert (tmp_path / "eval" / "metrics_summary.json").exists()


def test_evaluate_predictions_rejects_challenge_test_split(tmp_path: Path):
    label = np.zeros((5, 5, 5), dtype=np.uint8)
    pred = np.zeros((5, 5, 5), dtype=np.uint8)
    label[2, 2, 2] = 1
    pred[2, 2, 2] = 1

    label_path = tmp_path / "case001_wmh.npy"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred_path = pred_dir / "case001_pred.npy"

    np.save(label_path, label)
    np.save(pred_path, pred)

    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "test",
                "site": "fixture",
                "scanner_code": "fixture",
                "wmh_path": str(label_path),
            }
        ]
    )
    split = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "test",
                "source_split": "test",
                "assigned_split": "val",
                "site": "fixture",
                "scanner_code": "fixture",
            }
        ]
    )
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    import pytest

    with pytest.raises(ValueError, match=r"challenge_split=test|test split"):
        evaluate_predictions(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            prediction_dir=pred_dir,
            out_dir=tmp_path / "eval",
            run_id="fixture_run",
            assigned_split="val",
        )


def test_image_io_npy_roundtrip(tmp_path: Path):
    ref = tmp_path / "ref.npy"
    out = tmp_path / "out.npy"
    arr = np.ones((2, 2, 2), dtype=np.uint8)
    np.save(ref, arr)
    save_array_like(ref, out, arr)
    loaded = load_array(out)
    assert loaded.shape == (2, 2, 2)
    assert loaded.sum() == 8


def test_evaluate_predictions_rejects_shape_mismatch(tmp_path: Path):
    label = np.zeros((5, 5, 5), dtype=np.uint8)
    pred = np.zeros((4, 5, 5), dtype=np.uint8)

    label_path = tmp_path / "case001_wmh.npy"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred_path = pred_dir / "case001_pred.npy"

    np.save(label_path, label)
    np.save(pred_path, pred)

    manifest = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "site": "fixture",
                "scanner_code": "fixture",
                "wmh_path": str(label_path),
            }
        ]
    )
    split = pd.DataFrame(
        [
            {
                "case_id": "case001",
                "challenge_split": "training",
                "source_split": "training",
                "assigned_split": "val",
                "site": "fixture",
                "scanner_code": "fixture",
            }
        ]
    )
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_predictions(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            prediction_dir=pred_dir,
            out_dir=tmp_path / "eval",
            run_id="fixture_run",
            assigned_split="val",
        )


def test_evaluate_predictions_records_shape_metadata(tmp_path: Path):
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    label[1, 1, 1] = 1
    pred[1, 1, 1] = 1

    label_path = tmp_path / "case001_wmh.npy"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred_path = pred_dir / "case001_pred.npy"

    np.save(label_path, label)
    np.save(pred_path, pred)

    manifest = pd.DataFrame([{"case_id": "case001", "wmh_path": str(label_path)}])
    split = pd.DataFrame([{"case_id": "case001", "assigned_split": "val"}])
    manifest_csv = tmp_path / "manifest.csv"
    split_csv = tmp_path / "split.csv"
    manifest.to_csv(manifest_csv, index=False)
    split.to_csv(split_csv, index=False)

    out = evaluate_predictions(
        manifest_csv=manifest_csv,
        split_csv=split_csv,
        prediction_dir=pred_dir,
        out_dir=tmp_path / "eval",
        run_id="fixture_run",
        assigned_split="val",
    )

    case_metrics = pd.read_csv(out["case_metrics_csv"])
    assert case_metrics.loc[0, "prediction_shape"] == "3x3x3"
    assert case_metrics.loc[0, "label_shape"] == "3x3x3"
    assert case_metrics.loc[0, "geometry_policy"] == "shape+spacing+affine"
