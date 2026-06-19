from __future__ import annotations

from scripts.check_environment import (
    _check_wmh2017_root,
    _distribution_name,
    _is_doc_placeholder,
    _version_ok,
)


def test_distribution_name_for_submodule() -> None:
    assert _distribution_name("scipy.ndimage") == "scipy"


def test_is_doc_placeholder_detects_angle_brackets() -> None:
    assert _is_doc_placeholder("<LOCAL_WMH2017_FILES_ROOT>") is True
    assert _is_doc_placeholder("") is True
    assert _is_doc_placeholder("/data/wmh/files") is False


def test_version_ok_uses_root_distribution_for_submodule() -> None:
    ok, msg = _version_ok("scipy.ndimage", "1.13.1", {"scipy": "1.13.1"})
    assert ok is True
    assert msg == ""


def test_check_wmh2017_root_treats_doc_placeholder_as_unset() -> None:
    status, ok, warning = _check_wmh2017_root("<LOCAL_WMH2017_FILES_ROOT>")
    assert status == "unset"
    assert ok is False
    assert warning is None


def test_check_wmh2017_root_invalid_path_warns_not_import_failure(tmp_path) -> None:
    status, ok, warning = _check_wmh2017_root(str(tmp_path / "missing"))
    assert status == "invalid"
    assert ok is False
    assert warning is not None
