#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.evaluation.official_parity import ParityConfig, compare_official_parity


def _parse_tolerance(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --tolerance {item!r}; expected metric=value")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local WMH2017 case_metrics.csv against an official evaluator export. "
            "This script does not run or vendor official code."
        )
    )
    parser.add_argument("--local", required=True, help="Local case_metrics.csv from scripts/evaluate_wmh2017.py")
    parser.add_argument("--official", required=True, help="Official evaluator case-level CSV/TSV/JSON export")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--required-metric",
        action="append",
        default=[],
        help="Canonical metric required for parity. Repeatable. Defaults to dice, hd95, avd_percent, lesion_recall, lesion_f1.",
    )
    parser.add_argument("--tolerance", action="append", default=[], help="Override tolerance, e.g. dice=1e-6")
    parser.add_argument("--allow-missing-metrics", action="store_true")
    parser.add_argument("--allow-missing-cases", action="store_true")
    args = parser.parse_args()

    required = tuple(args.required_metric) if args.required_metric else (
        "dice",
        "hd95",
        "avd_percent",
        "lesion_recall",
        "lesion_f1",
    )
    report = compare_official_parity(
        local_metrics_path=args.local,
        official_metrics_path=args.official,
        out_dir=args.out_dir,
        config=ParityConfig(
            required_metrics=required,
            tolerances=_parse_tolerance(args.tolerance),
            allow_missing_metrics=args.allow_missing_metrics,
            allow_missing_cases=args.allow_missing_cases,
        ),
    )
    print(f"Wrote {Path(args.out_dir) / 'official_parity_report.json'}")
    print(f"status={report['status']} n_compared_cases={report['n_compared_cases']}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
