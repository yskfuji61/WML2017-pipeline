#!/usr/bin/env python3
"""Generate biweekly manager report (works with zero runs)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def _count_runs(repo_root: Path) -> list[str]:
    runs_dir = repo_root / "artifacts" / "runs"
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate biweekly report.")
    parser.add_argument("--period-start", required=True)
    parser.add_argument("--period-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    runs = _count_runs(repo_root)
    real_runs = [r for r in runs if not r.startswith("test_") and "preview" in r or "smoke" in r or "wmh2017" in r]

    lines = [
        f"# Biweekly report {args.period_start} to {args.period_end}",
        "",
        "## Scope",
        "Public WMH2017 local PoC technical verification only.",
        "",
        "## Run evidence",
    ]
    if not real_runs:
        lines.extend(
            [
                "- status: NO_REAL_RUN_EVIDENCE",
                "- note: No reviewed real run evidence in artifacts/runs for this period.",
            ]
        )
    else:
        lines.append(f"- run_ids_observed: {', '.join(real_runs[:10])}")
        lines.append("- status: local smoke / preview evidence may exist; verify lineage before claims")

    lines.extend(
        [
            "",
            "## Claim boundary",
            "- Allowed: local validation metric, public-data local PoC, technical verification",
            "- Prohibited: official benchmark, clinical/customer/production readiness, SOTA, READY_FOR_RELEASE",
            "",
            f"_generated_at: {datetime.now(timezone.utc).isoformat()}_",
        ]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
