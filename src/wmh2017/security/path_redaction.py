"""Redact local absolute paths before writing run evidence."""
from __future__ import annotations

from pathlib import Path


def redact_path(path: str | Path | None, *, home: Path | None = None) -> str:
    if path is None:
        return ""
    text = str(path)
    home = home or Path.home()
    try:
        rel = Path(text).resolve().relative_to(home.resolve())
        return f"~/{rel.as_posix()}"
    except ValueError:
        return "<redacted_absolute_path>"
