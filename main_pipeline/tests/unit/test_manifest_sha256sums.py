from pathlib import Path

from wmh2017.data.manifest import build_manifest, load_sha256sums


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_manifest_records_expected_sha256_without_hashing_raw_files(tmp_path):
    files = tmp_path / "files"
    case_dir = files / "training" / "Utrecht" / "0"
    _touch(case_dir / "pre" / "FLAIR.nii.gz")
    _touch(case_dir / "pre" / "T1.nii.gz")
    _touch(case_dir / "wmh.nii.gz")

    sha_path = tmp_path / "SHA256SUMS.txt"
    sha_path.write_text(
        "\n".join(
            [
                "a" * 64 + "  files/training/Utrecht/0/pre/FLAIR.nii.gz",
                "b" * 64 + "  files/training/Utrecht/0/pre/T1.nii.gz",
                "c" * 64 + "  files/training/Utrecht/0/wmh.nii.gz",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    df = build_manifest(files, sha256sums=load_sha256sums(sha_path))
    row = df.iloc[0]

    assert row["flair_expected_sha256"] == "a" * 64
    assert row["t1_expected_sha256"] == "b" * 64
    assert row["mask_expected_sha256"] == "c" * 64
    assert row["flair_sha256"] == ""
    assert row["mask_sha256"] == ""
    assert row["challenge_split"] == "training"
    assert row["scanner_code"] == "utrecht_philips_achieva_3t"
