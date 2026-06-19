"""Minimal pipeline fixture test (synthetic, no real WMH2017 root)."""

from __future__ import annotations

import json
from pathlib import Path

from wmh2017.evidence import materialize_v4_artifacts, record_failed_run


def test_record_failed_run_writes_status(tmp_path: Path):
    run_dir = tmp_path / "run"
    record_failed_run(run_dir, run_id="test_failed", seed=20260616, reason="synthetic_failure")
    payload = json.loads((run_dir / "failed_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED_RUN"
    assert (run_dir / "evidence_summary.md").exists()


def test_materialize_v4_artifacts_minimal(tmp_path: Path):
    run_dir = tmp_path / "run_ok"
    run_dir.mkdir(parents=True)
    (run_dir / "run_context.json").write_text('{"dataset_manifest_sha256":"abc"}', encoding="utf-8")
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "metrics_summary.json").write_text(
        json.dumps({"run_id": "test", "dice_mean": 0.1, "claim_boundary": "local only"}),
        encoding="utf-8",
    )
    materialize_v4_artifacts(run_dir, run_id="test", seed=20260616, status="COMPLETED_OR_FAILED_RUN")
    assert (run_dir / "metrics_summary.json").exists()
    assert (run_dir / "evidence_summary.md").exists()
    text = (run_dir / "evidence_summary.md").read_text(encoding="utf-8")
    assert "public_data_local_poc_only" in text or "local validation" in text
