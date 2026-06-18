from pathlib import Path

import pytest

from scripts.verify_wmh2017_download_evidence import verify_download_evidence


def _write_minimal_evidence(root: Path, *, ok_line: str = "files/training/Utrecht/0/wmh.nii.gz: OK") -> None:
    root.mkdir()
    (root / "dataverse_metadata.json").write_text("{}", encoding="utf-8")
    (root / "download_manifest.tsv").write_text("id\tpath\n", encoding="utf-8")
    (root / "download_record.txt").write_text(
        "\n".join(
            [
                "Source DOI: 10.34894/AECRSD",
                "Number of downloaded files:",
                "    1 downloaded_file_manifest.txt",
                "Number of SHA256 entries:",
                "    1 SHA256SUMS.txt",
                "All file sizes match Dataverse metadata.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "downloaded_file_manifest.txt").write_text("files/training/Utrecht/0/wmh.nii.gz\n", encoding="utf-8")
    (root / "SHA256SUMS.txt").write_text("abc  files/training/Utrecht/0/wmh.nii.gz\n", encoding="utf-8")
    (root / "sha256_verify.log").write_text(ok_line + "\n", encoding="utf-8")
    (root / "readme.pdf").write_bytes(b"%PDF-1.4\n")


def test_verify_download_evidence_passes_minimal_fixture(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_minimal_evidence(evidence)

    result = verify_download_evidence(evidence)

    assert result["status"] == "passed"
    assert result["downloaded_file_count"] == 1
    assert result["raw_medical_files_in_evidence_package"] == 0


def test_verify_download_evidence_rejects_failed_sha_line(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_minimal_evidence(evidence, ok_line="files/training/Utrecht/0/wmh.nii.gz: FAILED")

    with pytest.raises(ValueError, match="sha256 verification failures"):
        verify_download_evidence(evidence)


def test_verify_download_evidence_rejects_packaged_raw_medical_image(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_minimal_evidence(evidence)
    (evidence / "raw.nii.gz").write_bytes(b"not really nifti")

    with pytest.raises(ValueError, match="raw medical image files"):
        verify_download_evidence(evidence)
