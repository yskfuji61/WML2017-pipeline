#!/usr/bin/env python3
"""Refuse raw medical files under mlruns/."""

from __future__ import annotations

import argparse
from pathlib import Path

FORBIDDEN_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".nrrd", ".mha", ".mhd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mlruns has no raw medical files.")
    parser.add_argument("mlruns_dir", nargs="?", default="mlruns")
    args = parser.parse_args()

    root = Path(args.mlruns_dir)
    if not root.exists():
        print("mlruns raw-data gate PASS (no mlruns dir)")
        return

    offenders = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().endswith(FORBIDDEN_SUFFIXES):
            offenders.append(path.as_posix())
    if offenders:
        raise SystemExit("mlruns raw-data gate FAIL:\n" + "\n".join(offenders[:20]))
    print("mlruns raw-data gate PASS")


if __name__ == "__main__":
    main()
