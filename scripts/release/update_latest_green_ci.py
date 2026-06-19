#!/usr/bin/env python3
"""Record latest green CI URL for GAP-007 (Wave 3)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    out = Path("docs/release_evidence/latest_green_ci.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    run_url = os.environ.get("GITHUB_RUN_URL", "PENDING_CONFIRMATION")
    workflow = os.environ.get("GITHUB_WORKFLOW", "PENDING_CONFIRMATION")
    sha = os.environ.get("GITHUB_SHA", "PENDING_CONFIRMATION")
    lines = [
        "# Latest green CI",
        "",
        f"- recorded_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- workflow: {workflow}",
        f"- commit: {sha}",
        f"- run_url: {run_url}",
        "",
        "Manual verification required when run_url is PENDING_CONFIRMATION.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
