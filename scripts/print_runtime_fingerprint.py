"""Write runtime fingerprint JSON for a run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.lineage.runtime_fingerprint import write_runtime_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Write runtime_fingerprint.json for a run.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    payload = write_runtime_fingerprint(args.out, repo_root=args.repo_root)
    print(f"Wrote {args.out}")
    print(f"git_commit={payload.get('git_commit')}")


if __name__ == "__main__":
    main()
