#!/usr/bin/env python3
"""Generate evidence summary markdown for a run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wmh2017.evidence import write_evidence_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence_summary.md for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--status", default="COMPLETED_OR_FAILED_RUN")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_id = args.run_id or run_dir.name
    out = write_evidence_summary(run_dir, run_id=run_id, status=args.status)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
