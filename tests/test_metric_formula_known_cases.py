def test_formula_lock_doc_exists():
    from pathlib import Path

    spec = Path(__file__).resolve().parents[1] / "docs/model_validation/metric_formula_spec.md"
    assert "dice_local" in spec.read_text(encoding="utf-8")
