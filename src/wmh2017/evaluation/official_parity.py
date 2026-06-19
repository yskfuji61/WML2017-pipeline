"""Compare local WMH2017 validation metrics with an official evaluator export.

This module intentionally does not vendor or execute third-party challenge code.
It consumes an explicit official-evaluator result file supplied by a human/operator
and verifies that local metrics are numerically compatible within configured
tolerances before any leaderboard-comparable claim is made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh2017.audit.run_record import sha256_path

CASE_ID_ALIASES = ("case_id", "case", "subject", "subject_id", "id")
DEFAULT_METRIC_ALIASES = {
    "dice": ("dice", "dsc", "Dice", "DSC"),
    "hd95": ("hd95", "h95", "H95", "HD95", "hausdorff95", "Hausdorff95"),
    "avd_percent": ("avd_percent", "avd", "AVD", "absolute_volume_difference", "absolute_volume_difference_percent"),
    "lesion_recall": ("lesion_recall", "recall", "Recall", "lesionRecall"),
    "lesion_f1": ("lesion_f1", "f1", "F1", "lesionF1"),
}
DEFAULT_TOLERANCES = {
    "dice": 1e-6,
    "hd95": 1e-6,
    "avd_percent": 1e-6,
    "lesion_recall": 1e-6,
    "lesion_f1": 1e-6,
}


@dataclass(frozen=True)
class ParityConfig:
    required_metrics: tuple[str, ...] = ("dice", "hd95", "avd_percent", "lesion_recall", "lesion_f1")
    tolerances: dict[str, float] | None = None
    allow_missing_metrics: bool = False
    allow_missing_cases: bool = False


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metrics table not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("case_metrics", "cases", "rows", "results"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
    raise ValueError(f"unsupported metrics table format: {p}; expected .csv, .tsv, .txt or .json")


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return str(alias)
        found = lowered.get(alias.lower())
        if found is not None:
            return found
    return None


def _canonicalize_case_metrics(df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    case_col = _find_column(df, CASE_ID_ALIASES)
    if case_col is None:
        raise ValueError(f"{table_name} metrics must contain a case id column; accepted aliases={CASE_ID_ALIASES}")

    out = pd.DataFrame({"case_id": df[case_col].astype(str)})
    for canonical, aliases in DEFAULT_METRIC_ALIASES.items():
        col = _find_column(df, aliases)
        if col is not None:
            out[canonical] = pd.to_numeric(df[col], errors="coerce")
    return out


def compare_official_parity(
    local_metrics_path: str | Path,
    official_metrics_path: str | Path,
    out_dir: str | Path,
    *,
    config: ParityConfig | None = None,
) -> dict[str, Any]:
    """Compare local and official case-level metrics.

    Returns a JSON-serializable report and writes:
    - official_parity_case_diffs.csv
    - official_parity_report.json
    """
    cfg = config or ParityConfig()
    tolerances = {**DEFAULT_TOLERANCES, **(cfg.tolerances or {})}

    local_raw = _read_table(local_metrics_path)
    official_raw = _read_table(official_metrics_path)
    local = _canonicalize_case_metrics(local_raw, table_name="local")
    official = _canonicalize_case_metrics(official_raw, table_name="official")

    missing_metrics = [m for m in cfg.required_metrics if m not in local.columns or m not in official.columns]
    if missing_metrics and not cfg.allow_missing_metrics:
        raise ValueError(f"required metrics missing from local or official table: {missing_metrics}")

    comparable_metrics = [m for m in cfg.required_metrics if m in local.columns and m in official.columns]
    if not comparable_metrics:
        raise ValueError("no comparable metrics found between local and official tables")

    merged = local.merge(official, on="case_id", how="outer", suffixes=("_local", "_official"), indicator=True)
    missing_cases = merged[merged["_merge"] != "both"]["case_id"].astype(str).tolist()
    if missing_cases and not cfg.allow_missing_cases:
        raise ValueError(f"case mismatch between local and official metrics: {missing_cases[:20]}")

    records: list[dict[str, Any]] = []
    for _, row in merged[merged["_merge"] == "both"].iterrows():
        record: dict[str, Any] = {"case_id": str(row["case_id"])}
        for metric in comparable_metrics:
            local_value = row[f"{metric}_local"]
            official_value = row[f"{metric}_official"]
            diff = (
                abs(float(local_value) - float(official_value))
                if pd.notna(local_value) and pd.notna(official_value)
                else np.inf
            )
            tolerance = float(tolerances.get(metric, 1e-6))
            record[f"{metric}_local"] = float(local_value) if pd.notna(local_value) else np.nan
            record[f"{metric}_official"] = float(official_value) if pd.notna(official_value) else np.nan
            record[f"{metric}_abs_diff"] = float(diff)
            record[f"{metric}_within_tolerance"] = bool(diff <= tolerance)
        records.append(record)

    diff_df = pd.DataFrame(records)
    metric_summaries = {}
    failed_metrics = []
    for metric in comparable_metrics:
        diff_col = f"{metric}_abs_diff"
        ok_col = f"{metric}_within_tolerance"
        max_abs_diff = (
            float(diff_df[diff_col].replace([np.inf, -np.inf], np.nan).max()) if not diff_df.empty else np.inf
        )
        all_within = bool(diff_df[ok_col].all()) if ok_col in diff_df.columns else False
        metric_summaries[metric] = {
            "tolerance": float(tolerances.get(metric, 1e-6)),
            "max_abs_diff": max_abs_diff,
            "all_within_tolerance": all_within,
        }
        if not all_within:
            failed_metrics.append(metric)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    diff_csv = out / "official_parity_case_diffs.csv"
    diff_df.to_csv(diff_csv, index=False)

    report = {
        "status": "passed" if not failed_metrics and not missing_cases else "failed",
        "local_metrics_path": str(local_metrics_path),
        "local_metrics_sha256": sha256_path(local_metrics_path),
        "official_metrics_path": str(official_metrics_path),
        "official_metrics_sha256": sha256_path(official_metrics_path),
        "case_diff_csv": str(diff_csv),
        "case_diff_csv_sha256": sha256_path(diff_csv),
        "n_local_cases": int(len(local)),
        "n_official_cases": int(len(official)),
        "n_compared_cases": int(len(diff_df)),
        "comparable_metrics": comparable_metrics,
        "missing_metrics": missing_metrics,
        "missing_cases": missing_cases,
        "metric_summaries": metric_summaries,
        "claim_boundary": (
            "parity passing only means this local evaluator matched the supplied official export; "
            "it is not a clinical, customer, commercial, SOTA, or leaderboard claim without human review"
        ),
    }
    report_json = out / "official_parity_report.json"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
