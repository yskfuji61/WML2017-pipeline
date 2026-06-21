"""Split generation utilities with WMH2017 test-contamination guards."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "heldout", "heldout_eval"}


@dataclass(frozen=True)
class SplitPolicy:
    split_id: str
    seed: int = 42
    train_ratio: float = 0.8
    source_split_col: str = "challenge_split"


def normalize_source_split(value: object) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if s in TRAIN_NAMES:
        return "training"
    if s in TEST_NAMES:
        return "test"
    return s or "unknown"


def _resolve_source_split_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "challenge_split" in df.columns:
        return "challenge_split"
    if "source_split" in df.columns:
        return "source_split"
    raise ValueError("manifest must include challenge_split or source_split")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def make_train_val_split(
    manifest: pd.DataFrame,
    policy: SplitPolicy = SplitPolicy(split_id="WMH2017-TRAIN-VAL-SEED42"),
) -> pd.DataFrame:
    """Create train/val split from challenge training rows only.

    Critical rule:
    `test` rows are assigned to `heldout_eval` and are never eligible for train,
    validation, threshold tuning, preprocessing fit, model selection, or early
    stopping, even when `has_wmh` or `wmh_path` is present in a 2022 local release.
    """
    if "case_id" not in manifest.columns:
        raise ValueError("manifest must include case_id")

    source_col = _resolve_source_split_col(manifest, policy.source_split_col)
    df = manifest.copy()
    df["_normalized_source_split"] = df[source_col].map(normalize_source_split)

    # Only challenge training rows can be used for train/val.
    training = df[df["_normalized_source_split"] == "training"].copy()
    heldout = df[df["_normalized_source_split"] == "test"].copy()
    unknown = df[~df["_normalized_source_split"].isin(["training", "test"])].copy()

    # Optional, conservative guard: if has_wmh exists, require a primary reference for train/val.
    if "has_wmh" in training.columns:
        training = training[training["has_wmh"].map(_truthy)].copy()

    if training.empty:
        raise ValueError("No eligible challenge training rows with primary WMH reference found.")

    train_cases = (
        training[["case_id"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=policy.seed)["case_id"]
        .tolist()
    )
    n_train = max(1, int(round(len(train_cases) * policy.train_ratio)))
    if len(train_cases) > 1:
        n_train = min(n_train, len(train_cases) - 1)
    train_set = set(train_cases[:n_train])

    rows = []
    for _, r in training.iterrows():
        rows.append({
            "split_id": policy.split_id,
            "case_id": r["case_id"],
            "challenge_split": "training",
            "source_split": "training",
            "assigned_split": "train" if r["case_id"] in train_set else "val",
            "site": r.get("site", ""),
            "scanner": r.get("scanner", ""),
            "scanner_code": r.get("scanner_code", ""),
            "group_id": r.get("case_id", ""),
            "seed": policy.seed,
            "reason": "challenge_training_random_seeded_primary_reference",
        })
    for _, r in heldout.iterrows():
        rows.append({
            "split_id": policy.split_id,
            "case_id": r["case_id"],
            "challenge_split": "test",
            "source_split": "test",
            "assigned_split": "heldout_eval",
            "site": r.get("site", ""),
            "scanner": r.get("scanner", ""),
            "scanner_code": r.get("scanner_code", ""),
            "group_id": r.get("case_id", ""),
            "seed": policy.seed,
            "reason": "challenge_test_never_train_val_even_if_mask_exists",
        })
    for _, r in unknown.iterrows():
        rows.append({
            "split_id": policy.split_id,
            "case_id": r["case_id"],
            "challenge_split": r["_normalized_source_split"],
            "source_split": r["_normalized_source_split"],
            "assigned_split": "blocked_unknown_source_split",
            "site": r.get("site", ""),
            "scanner": r.get("scanner", ""),
            "scanner_code": r.get("scanner_code", ""),
            "group_id": r.get("case_id", ""),
            "seed": policy.seed,
            "reason": "unknown_source_split_requires_review",
        })
    out = pd.DataFrame(rows)
    out["created_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return out


@dataclass(frozen=True)
class KFoldPolicy:
    split_id: str
    k: int = 5
    seed: int = 42
    stratify_col: str = "site"
    source_split_col: str = "challenge_split"


def _stratified_fold_of(training: pd.DataFrame, *, k: int, seed: int, stratify_col: str) -> dict[str, int]:
    """Assign each training case_id to exactly one fold, stratified by stratify_col.

    Cases within a stratum are shuffled deterministically (by seed) and dealt
    round-robin across folds, so each fold receives a balanced share of every
    stratum. Grouping is by case_id, so a case appears in exactly one validation fold.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for cross-validation")
    strat = stratify_col if stratify_col in training.columns else None
    fold_of: dict[str, int] = {}
    if strat is None:
        cases = training[["case_id"]].drop_duplicates().sample(frac=1.0, random_state=seed)["case_id"].tolist()
        for j, case_id in enumerate(cases):
            fold_of[str(case_id)] = j % k
        return fold_of
    # Deterministic per-stratum dealing; continue the counter across strata so that
    # uneven stratum sizes still produce balanced overall fold sizes.
    counter = 0
    for stratum in sorted(training[strat].astype(str).fillna("").unique().tolist()):
        sub = training[training[strat].astype(str).fillna("") == stratum]
        cases = sub[["case_id"]].drop_duplicates().sample(frac=1.0, random_state=seed)["case_id"].tolist()
        for case_id in cases:
            fold_of[str(case_id)] = counter % k
            counter += 1
    return fold_of


def make_kfold_splits(
    manifest: pd.DataFrame,
    policy: KFoldPolicy = KFoldPolicy(split_id="WMH2017-KFOLD-SEED42"),
) -> list[pd.DataFrame]:
    """Create k site-stratified CV folds from challenge training rows only.

    Returns a list of k DataFrames. In fold i, the i-th stratified group is the
    validation set (assigned_split=val), all other training cases are train, and
    challenge test rows are always heldout_eval (never train/val), mirroring the
    test-contamination rule of make_train_val_split.
    """
    if "case_id" not in manifest.columns:
        raise ValueError("manifest must include case_id")

    source_col = _resolve_source_split_col(manifest, policy.source_split_col)
    df = manifest.copy()
    df["_normalized_source_split"] = df[source_col].map(normalize_source_split)

    training = df[df["_normalized_source_split"] == "training"].copy()
    heldout = df[df["_normalized_source_split"] == "test"].copy()

    if "has_wmh" in training.columns:
        training = training[training["has_wmh"].map(_truthy)].copy()
    if training.empty:
        raise ValueError("No eligible challenge training rows with primary WMH reference found.")

    fold_of = _stratified_fold_of(
        training, k=policy.k, seed=policy.seed, stratify_col=policy.stratify_col
    )

    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    folds: list[pd.DataFrame] = []
    for fold_idx in range(policy.k):
        rows: list[dict[str, object]] = []
        for _, r in training.iterrows():
            case_id = str(r["case_id"])
            assigned = "val" if fold_of[case_id] == fold_idx else "train"
            rows.append(
                {
                    "split_id": policy.split_id,
                    "fold": fold_idx,
                    "case_id": r["case_id"],
                    "challenge_split": "training",
                    "source_split": "training",
                    "assigned_split": assigned,
                    "site": r.get("site", ""),
                    "scanner": r.get("scanner", ""),
                    "scanner_code": r.get("scanner_code", ""),
                    "group_id": r.get("case_id", ""),
                    "seed": policy.seed,
                    "reason": "challenge_training_site_stratified_kfold",
                }
            )
        for _, r in heldout.iterrows():
            rows.append(
                {
                    "split_id": policy.split_id,
                    "fold": fold_idx,
                    "case_id": r["case_id"],
                    "challenge_split": "test",
                    "source_split": "test",
                    "assigned_split": "heldout_eval",
                    "site": r.get("site", ""),
                    "scanner": r.get("scanner", ""),
                    "scanner_code": r.get("scanner_code", ""),
                    "group_id": r.get("case_id", ""),
                    "seed": policy.seed,
                    "reason": "challenge_test_never_train_val_even_if_mask_exists",
                }
            )
        out = pd.DataFrame(rows)
        out["created_at"] = created_at
        folds.append(out)
    return folds


def assert_kfold_coverage(folds: list[pd.DataFrame]) -> None:
    """Assert val sets partition the training cases with no leakage or overlap."""
    if not folds:
        raise AssertionError("no folds provided")
    for f in folds:
        assert_no_test_contamination(f)

    train_cases: set[str] = set()
    val_union: set[str] = set()
    val_lists: list[set[str]] = []
    for f in folds:
        tv = f[f["assigned_split"].isin(["train", "val"])]
        train_cases.update(tv["case_id"].astype(str).tolist())
        vset = set(f[f["assigned_split"] == "val"]["case_id"].astype(str).tolist())
        val_lists.append(vset)
        val_union.update(vset)

    if val_union != train_cases:
        missing = train_cases - val_union
        raise AssertionError(f"validation folds do not cover all training cases; missing={sorted(missing)}")

    # Each case must appear in exactly one validation fold.
    seen: dict[str, int] = {}
    for i, vset in enumerate(val_lists):
        for case_id in vset:
            if case_id in seen:
                raise AssertionError(
                    f"case {case_id} appears in val folds {seen[case_id]} and {i}"
                )
            seen[case_id] = i


def assert_no_test_contamination(split_df: pd.DataFrame) -> None:
    source = split_df["source_split"].astype(str).str.lower()
    assigned = split_df["assigned_split"].astype(str).str.lower()
    bad = split_df[(source == "test") & (assigned.isin(["train", "val", "validation"]))]
    if not bad.empty:
        raise AssertionError(f"test split contamination detected: {bad['case_id'].tolist()}")
