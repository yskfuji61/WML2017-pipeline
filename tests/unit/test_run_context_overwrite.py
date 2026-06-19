from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from wmh2017.lineage.run_context import clear_run_work_dir, init_run_directory


def test_clear_run_work_dir_removes_gitignored_run_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "artifacts" / "runs" / "wmh2017_full_short_seed42"
    run_dir.mkdir(parents=True)
    marker = run_dir / "run_context.json"
    marker.write_text("{}", encoding="utf-8")

    removed = clear_run_work_dir(run_dir, repo_root)

    assert removed is True
    assert not run_dir.exists()


def test_clear_run_work_dir_refuses_paths_outside_artifacts_runs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    outside = repo_root / "reports" / "runs" / "bad"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="refuse to overwrite"):
        clear_run_work_dir(outside, repo_root)


def test_init_run_directory_after_clear_allows_rerun(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "artifacts" / "runs" / "rerun"
    run_dir.mkdir(parents=True)
    (run_dir / "stale.txt").write_text("old", encoding="utf-8")

    clear_run_work_dir(run_dir, repo_root)
    with patch(
        "wmh2017.lineage.run_context.build_run_context",
        return_value={"run_id": "rerun"},
    ):
        init_run_directory(run_dir, run_id="rerun", wmh2017_root="/data/files", seed=42)

    assert (run_dir / "run_context.json").exists()
    assert not (run_dir / "stale.txt").exists()
