#!/usr/bin/env python3
"""CLI gate: active code must not import legacy core/pipeline modules."""

from __future__ import annotations

import argparse
from pathlib import Path

ACTIVE_ROOTS = [Path("src/wmh2017"), Path("scripts")]
FORBIDDEN = ["core.pipeline", "from core", "import core"]


def check_legacy_imports(repo_root: Path) -> list[str]:
    offenders: list[str] = []
    skip_names = {"verify_no_legacy_imports.py"}
    for root_rel in ACTIVE_ROOTS:
        root = repo_root / root_rel
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name in skip_names:
                continue
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                if token in text:
                    offenders.append(f"{path.relative_to(repo_root)} imports legacy token: {token}")
    return offenders


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify active code does not import legacy modules.")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    offenders = check_legacy_imports(repo_root)
    if offenders:
        raise SystemExit("legacy import gate FAIL:\n" + "\n".join(offenders))
    print("legacy import gate PASS")


if __name__ == "__main__":
    main()
