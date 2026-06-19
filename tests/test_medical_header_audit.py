from pathlib import Path

from wmh2017.medical_header_audit import audit_file, audit_root, redact_path


def test_redact_path_never_returns_absolute():
    assert redact_path(Path("/Users/x/file.nii.gz")) == "REDACTED_OR_LOCAL_ONLY"


def test_unsupported_format_marked():
    result = audit_file(Path("sample.txt"))
    assert result.status == "UNSUPPORTED_FORMAT"


def test_audit_root_missing_is_not_success(tmp_path):
    payload = audit_root(tmp_path, max_files=1)
    assert payload["raw_metadata_values_included"] is False
    assert "audit_hash" in payload
