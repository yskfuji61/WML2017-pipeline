"""e2e configured-split loader: opt-in CV fold support, backward compatible."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh2017.e2e.context import E2ERunContext
from wmh2017.e2e.stages import _configured_split_to_load


def _ctx(config_path: Path, repo_root: Path) -> E2ERunContext:
    return E2ERunContext(
        repo_root=repo_root,
        run_id="t",
        files_root="x",
        seed=42,
        work_dir=repo_root / "work",
        config_path=config_path,
        sha256sums="",
        official_metrics="",
        skip_train=False,
        no_inspect_images=True,
        allow_dirty_git=True,
    )


def test_default_config_regenerates_split(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data:\n  split_manifest: data/splits/x.csv\n", encoding="utf-8")
    assert _configured_split_to_load(_ctx(cfg, tmp_path)) is None


def test_use_configured_split_loads_existing(tmp_path: Path) -> None:
    split = tmp_path / "data/splits/fold0.csv"
    split.parent.mkdir(parents=True)
    split.write_text("case_id,assigned_split\nc1,train\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "data:\n  use_configured_split: true\n  split_manifest: data/splits/fold0.csv\n",
        encoding="utf-8",
    )
    resolved = _configured_split_to_load(_ctx(cfg, tmp_path))
    assert resolved == split


def test_use_configured_split_missing_file_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "data:\n  use_configured_split: true\n  split_manifest: data/splits/missing.csv\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        _configured_split_to_load(_ctx(cfg, tmp_path))
