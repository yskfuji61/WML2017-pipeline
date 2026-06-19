"""Human-readable failure summaries for pipeline stages."""

from __future__ import annotations


def failure_summary(*, stage: str, error: str) -> str:
    return (
        f"Stage '{stage}' failed: {error}. "
        "Inspect artifacts/runs/<run_id>/command_log.jsonl, fix the root cause, "
        "commit changes, and rerun with a clean git tree or --allow-dirty-git."
    )
