import subprocess
from pathlib import Path


def test_git_tracked_files_do_not_include_nifti_suffixes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=str(repo_root), text=True).splitlines()
    forbidden = (".nii", ".nii.gz", ".dcm")
    offenders = [p for p in tracked if p.lower().endswith(forbidden)]
    assert not offenders, f"tracked raw medical files found: {offenders}"
