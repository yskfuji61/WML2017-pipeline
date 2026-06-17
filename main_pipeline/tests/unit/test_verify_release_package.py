from pathlib import Path

import pytest

from scripts.verify_release_package import relative_to_root


def test_relative_to_root_rejects_outside_output(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside" / "manifest.json"

    with pytest.raises(SystemExit, match="must resolve inside --repo-root"):
        relative_to_root(outside, repo_root.resolve(), "--out")
