"""Write offline observability dashboard from run event logs."""

from __future__ import annotations

import json
from pathlib import Path


def write_offline_dashboard(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    out = output or run_dir / "observability" / "offline_dashboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    event_log = run_dir / "observability" / "event_log.jsonl"
    lines = ["# Offline run dashboard", "", "- run_dir: REDACTED_OR_LOCAL_ONLY", ""]
    if event_log.exists():
        lines.append("## Events")
        for row in event_log.read_text(encoding="utf-8").splitlines()[:50]:
            try:
                item = json.loads(row)
                lines.append(f"- {item.get('event', 'unknown')}: {item.get('run_id', '')}")
            except json.JSONDecodeError:
                continue
    else:
        lines.append("- status: NO_EVENT_LOG")

    summary = run_dir / "observability" / "offline_run_summary.json"
    if summary.exists():
        lines.extend(["", "## Summary", "- present: true"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
