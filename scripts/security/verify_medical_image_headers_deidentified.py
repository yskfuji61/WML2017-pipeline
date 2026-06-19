#!/usr/bin/env python3
"""Redacted medical image header audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.medical_header_audit import audit_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit medical image headers with redacted output.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-files", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        payload = {"status": "MISSING_WMH2017_ROOT", "root": "REDACTED_OR_LOCAL_ONLY", "results": []}
    else:
        payload = audit_root(root, max_files=args.max_files)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
