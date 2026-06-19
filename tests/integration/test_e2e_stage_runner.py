from pathlib import Path
from unittest.mock import patch

import pytest

from wmh2017.e2e.context import E2ERunContext
from wmh2017.e2e.runner import run_pipeline


def _make_ctx(tmp_path: Path, **overrides: object) -> E2ERunContext:
    repo_root = Path(__file__).resolve().parents[2]
    defaults = {
        "repo_root": repo_root,
        "run_id": "test_run",
        "files_root": str(tmp_path / "files"),
        "seed": 42,
        "work_dir": tmp_path / "run",
        "config_path": repo_root / "configs/wmh2017_monai_smoke.yaml",
        "sha256sums": "evidence/wmh2017_download_2026-06-16/SHA256SUMS.txt",
        "official_metrics": "",
        "skip_train": False,
        "no_inspect_images": True,
        "allow_dirty_git": True,
    }
    defaults.update(overrides)
    return E2ERunContext(**defaults)


def _fake_run_factory(calls: list[list[str]]):
    def fake_run(cmd, *, cwd):
        calls.append(list(cmd))
        script = cmd[1] if len(cmd) > 1 else ""
        if "audit_wmh2017_dataset.py" in script and "--out" in cmd:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("dataset_id,case_id\nWMH2017,case001\n", encoding="utf-8")
        elif "audit_wmh2017_labels.py" in script and "--out" in cmd:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("case_id,status\n", encoding="utf-8")
        elif "make_wmh2017_splits.py" in script and "--out-dir" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            seed = cmd[cmd.index("--seed") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            split_csv = out_dir / f"wmh2017_train_val_seed{seed}.csv"
            split_csv.write_text("split_id,case_id\n", encoding="utf-8")
        return {"cmd": list(cmd), "returncode": 0, "stdout": "", "stderr": ""}

    return fake_run


def _run_with_mocks(ctx: E2ERunContext, fake_run):
    with patch("wmh2017.e2e.stages.run_command", side_effect=fake_run):
        with patch("wmh2017.e2e.stages.copy_to_nested"):
            with patch("wmh2017.e2e.stages.write_hash_sidecar"):
                with patch("wmh2017.lineage.run_context.init_run_directory"):
                    with patch("wmh2017.lineage.runtime_fingerprint.write_runtime_fingerprint"):
                        with patch("wmh2017.e2e.runner.git_dirty", return_value=False):
                            return run_pipeline(ctx)


def test_run_pipeline_calls_dataset_audit_first(tmp_path: Path):
    ctx = _make_ctx(tmp_path, skip_train=True)
    calls: list[list[str]] = []
    _run_with_mocks(ctx, _fake_run_factory(calls))
    assert calls
    assert "scripts/audit_wmh2017_dataset.py" in calls[0][1]


def test_run_pipeline_skips_train_stages_when_requested(tmp_path: Path):
    ctx = _make_ctx(tmp_path, skip_train=True)
    calls: list[list[str]] = []
    result = _run_with_mocks(ctx, _fake_run_factory(calls))
    joined = " ".join(" ".join(cmd) for cmd in calls)
    assert "train_wmh2017.py" not in joined
    assert "evaluate_wmh2017.py" not in joined
    assert result.stage_status.get("training") is None


def test_run_pipeline_rejects_dirty_git_without_flag(tmp_path: Path):
    ctx = _make_ctx(tmp_path, allow_dirty_git=False, skip_train=True)
    with patch("wmh2017.lineage.run_context.init_run_directory"):
        with patch("wmh2017.lineage.runtime_fingerprint.write_runtime_fingerprint"):
            with patch("wmh2017.e2e.runner.git_dirty", return_value=True):
                with pytest.raises(SystemExit, match="dirty"):
                    run_pipeline(ctx)
