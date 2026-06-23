"""Case row resolution from dataset/split manifests (modality-aware).

This is the single authority that maps a split assignment to concrete image/label
paths. It replaces the per-module ``_case_rows`` helpers and enforces the
``challenge_split=test`` guard so the test split is never consumed for training,
validation, or threshold tuning.

Backward compatibility: with the legacy single ``"image"`` modality, resolution and
the emitted MONAI row dicts (``{"case_id", "image", "label"}``) are identical to the
prior ``train_monai._case_rows`` behavior, including the FLAIR path fallback chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wmh2017.config.training_config import LEGACY_IMAGE_NAME, InputModality

# Legacy fallback columns tried (in order) when the configured manifest key is empty.
# Only applied to the legacy single-channel "image" modality to preserve behavior.
_LEGACY_IMAGE_FALLBACK_KEYS = ("flair_path", "flair_pre_path")
_LABEL_FALLBACK_KEYS = ("wmh_path", "mask_path")


@dataclass(frozen=True)
class CaseRecord:
    """One resolved case: per-modality image paths plus the label path."""

    case_id: str
    image_paths: dict[str, str]
    label_path: str
    challenge_split: str


def _cell(row: pd.Series, key: str) -> str:
    """Read a manifest cell as a clean string, treating NaN/missing as empty."""
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _resolve_path(row: pd.Series, primary_key: str, fallback_keys: tuple[str, ...]) -> str:
    value = _cell(row, primary_key)
    if value:
        return value
    for key in fallback_keys:
        value = _cell(row, key)
        if value:
            return value
    return ""


def load_case_records(
    *,
    manifest_csv: str | Path,
    split_csv: str | Path,
    assigned_split: str,
    input_modalities: tuple[InputModality, ...],
    label_key: str,
) -> list[CaseRecord]:
    """Resolve case records for ``assigned_split``; raise on any test-split case."""
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    if split.empty:
        raise ValueError(f"no rows for assigned_split={assigned_split} in {split_csv}")

    records: list[CaseRecord] = []
    for _, s in split.iterrows():
        case_id = str(s["case_id"])
        m = manifest[manifest["case_id"].astype(str) == case_id]
        if m.empty:
            raise ValueError(f"case_id={case_id} exists in split but not manifest")
        r = m.iloc[0]
        challenge_split = str(r.get("challenge_split", ""))
        if challenge_split.lower() == "test":
            raise ValueError(f"case_id={case_id} belongs to challenge_split=test; cannot be used here")

        image_paths: dict[str, str] = {}
        for modality in input_modalities:
            fallback = _LEGACY_IMAGE_FALLBACK_KEYS if modality.name == LEGACY_IMAGE_NAME else ()
            path = _resolve_path(r, modality.manifest_key, fallback)
            if not path and modality.required:
                raise ValueError(
                    f"case_id={case_id} missing required modality '{modality.name}' "
                    f"(manifest_key={modality.manifest_key})"
                )
            image_paths[modality.name] = path

        label_path = _resolve_path(r, label_key, _LABEL_FALLBACK_KEYS)
        if not label_path:
            raise ValueError(f"case_id={case_id} missing label path (label_key={label_key})")

        records.append(
            CaseRecord(
                case_id=case_id,
                image_paths=image_paths,
                label_path=label_path,
                challenge_split=challenge_split,
            )
        )
    return records


def case_records_to_monai_rows(records: list[CaseRecord]) -> list[dict[str, str]]:
    """Flatten case records into MONAI dict rows: ``{case_id, <modality>..., label}``.

    For the legacy single ``"image"`` modality this yields the historical
    ``{"case_id", "image", "label"}`` shape exactly.
    """
    rows: list[dict[str, str]] = []
    for record in records:
        row: dict[str, str] = {"case_id": record.case_id, "label": record.label_path}
        row.update(record.image_paths)
        rows.append(row)
    return rows
