#!/usr/bin/env python3
"""Select the best experiment across CV summaries using a pre-registered rule.

Reads ``cv_summary_*.json`` files (from cv_aggregate), applies a fixed
``candidate_selection`` rule, and writes a hashed selection artifact. Local
cross-validation only; never consumes test-split metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.model_selection import (
    load_candidate,
    rule_from_config,
    select_candidate,
    write_candidate_selection_artifact,
)


def _resolve_summary_paths(args: argparse.Namespace) -> list[Path]:
    if args.cv_summaries:
        return [Path(p) for p in args.cv_summaries]
    if args.cv_summary_dir and args.cv_ids:
        base = Path(args.cv_summary_dir)
        return [base / f"cv_summary_{cv_id}.json" for cv_id in args.cv_ids]
    raise SystemExit("provide --cv-summaries PATH... or --cv-summary-dir with --cv-ids")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run candidate selection (val-only, pre-registered rule).")
    parser.add_argument("--cv-summaries", nargs="+", help="explicit cv_summary_*.json paths")
    parser.add_argument("--cv-summary-dir", default="reports/cv", help="dir holding cv_summary_*.json")
    parser.add_argument("--cv-ids", nargs="+", help="cv_ids to load from --cv-summary-dir")
    parser.add_argument("--rule-config", required=True, help="YAML with a candidate_selection block")
    parser.add_argument("--out", default="reports/cv/candidate_selection_result.json")
    args = parser.parse_args()

    summary_paths = _resolve_summary_paths(args)
    candidates = [load_candidate(p) for p in summary_paths]
    rule = rule_from_config(yaml.safe_load(Path(args.rule_config).read_text(encoding="utf-8")))

    result = select_candidate(candidates, rule)
    payload = write_candidate_selection_artifact(args.out, result)

    print(f"Wrote {args.out}")
    selected = payload["selected"]
    print(
        f"selected={selected['cv_id']} "
        f"{rule.primary_metric}={selected['metrics'].get(rule.primary_metric):.6f} "
        f"(policy: {payload['selection_policy']})"
    )
    if payload["excluded"]:
        print(f"excluded={[e['cv_id'] for e in payload['excluded']]}")


if __name__ == "__main__":
    main()
