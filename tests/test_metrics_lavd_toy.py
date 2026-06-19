"""v4 lAVD candidate boundary: local AVD is not official lAVD."""

from pathlib import Path


def test_lavd_candidate_documented_as_not_official_parity():
    spec = Path(__file__).resolve().parents[1] / "docs/model_validation/metric_formula_spec.md"
    text = spec.read_text(encoding="utf-8")
    assert "lavd_wmh2017_compat_candidate" in text
    assert "WMH2017_COMPAT_CANDIDATE" in text
    assert "not official lAVD" in text.lower() or "LOCAL_ONLY" in text
