"""Spec-driven N-member probability ensembling for WMH2017 (val-only, frozen weights).

An :class:`EnsembleSpec` declares N members (each a probability-map directory + a convex
weight), a threshold, and optional post-processing. Weights/threshold are **frozen** in the
spec: evaluation never grid-searches them, so a result cannot be produced by tuning weights
to the reporting fold or the test split. The spec is validated (>=2 members, weights >=0
summing to 1, threshold in range) and per-case member shapes must agree before fusion.

Additive to the legacy 2-member tuner in ``cross_arch_ensemble``; that module is unchanged.
Local validation only; not SOTA/official/clinical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.evaluation.cross_arch_ensemble import load_probability_volume
from wmh2017.evaluation.lesion_metrics import lesion_recall_f1_wmh_label1
from wmh2017.evaluation.postprocess import PostProcessConfig, apply_post_process
from wmh2017.evaluation.threshold_sweep import _case_rows_for_split
from wmh2017.evaluation.voxel_metrics import avd_wmh_label1, dice_wmh_label1, hd95_wmh_label1
from wmh2017.io.images import load_array, load_image_metadata
from wmh2017.lineage.hashes import sha256_jsonable, write_json

_CLAIM_BOUNDARY = "local validation ensemble evaluation only; not SOTA/official/clinical/production"
_PROHIBITED_USE = ["test weight tuning", "SOTA claim", "clinical decision", "production deployment"]


@dataclass(frozen=True)
class EnsembleMember:
    """One ensemble member: a probability-map directory and its convex weight."""

    name: str
    probs_dir: str
    weight: float


@dataclass(frozen=True)
class EnsembleSpec:
    """Declarative, frozen ensemble configuration."""

    members: tuple[EnsembleMember, ...]
    threshold: float = 0.5
    post_process: PostProcessConfig | None = None
    weight_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError(f"ensemble requires at least 2 members, got {len(self.members)}")
        names = [m.name for m in self.members]
        if len(set(names)) != len(names):
            raise ValueError(f"ensemble member names must be unique, got {names}")
        if any(m.weight < 0 for m in self.members):
            raise ValueError(f"ensemble weights must be >= 0, got {[m.weight for m in self.members]}")
        weight_sum = sum(m.weight for m in self.members)
        if abs(weight_sum - 1.0) > self.weight_tolerance:
            raise ValueError(f"ensemble weights must sum to 1.0 (convex), got {weight_sum}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {self.threshold}")

    def weights(self) -> list[float]:
        return [m.weight for m in self.members]


def fuse_probability_maps_weighted(
    prob_maps: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted average of N same-shape probability maps."""
    if len(prob_maps) != len(weights):
        raise ValueError(f"prob_maps ({len(prob_maps)}) and weights ({len(weights)}) length mismatch")
    if not prob_maps:
        raise ValueError("at least one probability map is required")
    base_shape = prob_maps[0].shape
    for i, pm in enumerate(prob_maps):
        if pm.shape != base_shape:
            raise ValueError(f"member[{i}] shape {pm.shape} != member[0] shape {base_shape}")
    fused = np.zeros(base_shape, dtype=np.float64)
    for pm, w in zip(prob_maps, weights, strict=True):
        fused += float(w) * pm.astype(np.float64)
    return fused.astype(np.float32)


def spec_from_config(mapping: Mapping[str, Any]) -> EnsembleSpec:
    """Parse an ``ensemble:`` block (or its parent mapping) into an EnsembleSpec."""
    block = mapping.get("ensemble", mapping)
    threshold = float(block.get("threshold", 0.5))
    members = tuple(
        EnsembleMember(name=str(m["name"]), probs_dir=str(m["probs_dir"]), weight=float(m["weight"]))
        for m in block["members"]
    )
    post_process: PostProcessConfig | None = None
    raw_pp = block.get("post_process")
    if raw_pp:
        post_process = PostProcessConfig(
            threshold=threshold,
            min_component_size=int(raw_pp.get("min_component_size", 0)),
            adaptive_low_threshold=float(raw_pp.get("adaptive_low_threshold", 0.0)),
            adaptive_high_volume_voxels=int(raw_pp.get("adaptive_high_volume_voxels", 0)),
        )
    return EnsembleSpec(members=members, threshold=threshold, post_process=post_process)


def spec_to_payload(spec: EnsembleSpec) -> dict[str, Any]:
    """Serialize the spec for the evaluation artifact."""
    payload: dict[str, Any] = {
        "threshold": spec.threshold,
        "members": [{"name": m.name, "probs_dir": m.probs_dir, "weight": m.weight} for m in spec.members],
    }
    if spec.post_process is not None:
        payload["post_process"] = {
            "min_component_size": spec.post_process.min_component_size,
            "adaptive_low_threshold": spec.post_process.adaptive_low_threshold,
            "adaptive_high_volume_voxels": spec.post_process.adaptive_high_volume_voxels,
        }
    return payload


def _load_member_maps(spec: EnsembleSpec, case_id: str) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for member in spec.members:
        path = Path(member.probs_dir) / f"{case_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"member '{member.name}' missing prob map for case_id={case_id}: {path}")
        maps.append(load_probability_volume(path))
    return maps


def evaluate_ensemble(
    *,
    spec: EnsembleSpec,
    manifest_csv: str | Path,
    split_csv: str | Path,
    assigned_split: str = "val",
) -> dict[str, Any]:
    """Fuse member probabilities with the spec's frozen weights and evaluate (val only)."""
    pp = spec.post_process or PostProcessConfig(threshold=spec.threshold)
    weights = spec.weights()
    records: list[dict[str, float | str]] = []
    for row in _case_rows_for_split(manifest_csv, split_csv, assigned_split):
        case_id = row["case_id"]
        fused = fuse_probability_maps_weighted(_load_member_maps(spec, case_id), weights)
        label_mask = wmh_foreground_mask(load_array(row["label_path"])).astype(np.uint8)
        spacing_raw = load_image_metadata(row["label_path"]).spacing
        spacing = spacing_raw if spacing_raw and len(spacing_raw) >= 3 else None
        pred = apply_post_process(fused, pp)
        lesion = lesion_recall_f1_wmh_label1(pred, label_mask)
        records.append(
            {
                "case_id": case_id,
                "dice": float(dice_wmh_label1(pred, label_mask)),
                "hd95": float(hd95_wmh_label1(pred, label_mask, spacing=spacing)),
                "avd": float(avd_wmh_label1(pred, label_mask, spacing=spacing)),
                "lesion_recall": float(lesion["lesion_recall"]),
                "lesion_f1": float(lesion["lesion_f1"]),
            }
        )
    df = pd.DataFrame(records)
    return {
        "n_cases": int(len(df)),
        "assigned_split": assigned_split,
        "threshold": float(spec.threshold),
        "weights": weights,
        "member_names": [m.name for m in spec.members],
        "mean_dice": float(df["dice"].mean()),
        "mean_lesion_recall": float(df["lesion_recall"].mean()),
        "mean_lesion_f1": float(df["lesion_f1"].mean()),
        "mean_hd95": float(df["hd95"].mean()),
        "mean_avd": float(df["avd"].mean()),
        "per_case": df,
    }


def write_ensemble_evaluation_artifact(
    *,
    out_dir: str | Path,
    summary: dict[str, Any],
    spec: EnsembleSpec,
    run_id: str,
) -> dict[str, Any]:
    """Write per-case CSV + a hashed evaluation JSON recording the frozen spec."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = out_dir / "ensemble_eval_per_case.csv"
    per_case = summary.get("per_case")
    if isinstance(per_case, pd.DataFrame):
        per_case.to_csv(per_case_csv, index=False)

    payload = {k: v for k, v in summary.items() if k != "per_case"}
    payload.update(
        {
            "run_id": run_id,
            "spec": spec_to_payload(spec),
            "weights_frozen": True,
            "weights_frozen_note": "weights/threshold are pre-registered in the spec; never re-tuned on the reporting fold or test split",
            "allowed_use": "validation-only ensemble evaluation",
            "prohibited_use": _PROHIBITED_USE,
            "claim_boundary": _CLAIM_BOUNDARY,
            "per_case_csv": str(per_case_csv),
        }
    )
    payload["artifact_hash"] = sha256_jsonable(payload)
    write_json(out_dir / "ensemble_evaluation.json", payload)
    return payload
