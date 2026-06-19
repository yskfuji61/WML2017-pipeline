from __future__ import annotations

import json
from pathlib import Path

from wmh2017.audit.run_record import append_run_manifest, make_run_row, update_run_manifest_metric


def test_update_run_manifest_metric_sets_path_and_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.csv"
    metric_path = tmp_path / "evaluation" / "metrics_summary.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(json.dumps({"mean_dice": 0.1}), encoding="utf-8")

    append_run_manifest(
        make_run_row(
            run_id="run_a",
            run_purpose="test",
            config_path=str(tmp_path / "config.yaml"),
            dataset_manifest=str(tmp_path / "dataset.csv"),
            split_manifest=str(tmp_path / "split.csv"),
        ),
        manifest_path,
    )

    updated = update_run_manifest_metric("run_a", metric_path, manifest_path=manifest_path)
    assert updated is True

    import pandas as pd

    df = pd.read_csv(manifest_path)
    row = df[df["run_id"] == "run_a"].iloc[0]
    assert row["metric_json_path"] == str(metric_path)
    assert row["metric_json_hash"]


def test_update_run_manifest_metric_returns_false_for_missing_run(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.csv"
    append_run_manifest(
        make_run_row(
            run_id="run_a",
            run_purpose="test",
            config_path=str(tmp_path / "config.yaml"),
            dataset_manifest=str(tmp_path / "dataset.csv"),
            split_manifest=str(tmp_path / "split.csv"),
        ),
        manifest_path,
    )

    updated = update_run_manifest_metric(
        "missing_run",
        tmp_path / "metrics_summary.json",
        manifest_path=manifest_path,
    )
    assert updated is False
