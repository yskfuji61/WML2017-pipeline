from pathlib import Path


def test_active_wmh_package_does_not_import_legacy_core_pipeline() -> None:
    root = Path("src/wmh2017")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "core.pipeline" in text or "from core" in text:
            offenders.append(str(path))
    assert not offenders, f"Active package imports legacy modules: {offenders}"
