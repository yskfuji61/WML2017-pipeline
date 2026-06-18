from pathlib import Path

ACTIVE_ROOTS = [Path("src/wmh2017"), Path("scripts")]
FORBIDDEN = ["core.pipeline", "from core", "import core"]


def test_active_code_does_not_import_legacy_core() -> None:
    offenders: list[str] = []
    skip_names = {"verify_no_legacy_imports.py"}
    for root in ACTIVE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name in skip_names:
                continue
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                if token in text:
                    offenders.append(f"{path} imports legacy token: {token}")
    assert not offenders, f"Active package imports legacy modules: {offenders}"
