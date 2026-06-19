#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.data.label_policy import wmh_foreground_mask


def _load_volume(path: str, *, nib: object) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def _best_slice(fg: np.ndarray, fallback_shape: tuple[int, ...]) -> int:
    if fg.any():
        return int(np.argmax(fg.sum(axis=(0, 1))))
    return fallback_shape[2] // 2 if len(fallback_shape) > 2 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Save FLAIR + GT and optional prediction overlays for one case.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--prediction", default="", help="Optional prediction NIfTI path")
    parser.add_argument("--out", default="reports/overlays")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
    except ImportError as e:
        raise SystemExit("nibabel and matplotlib are required for visualization.") from e

    df = pd.read_csv(args.manifest)
    row = df[df["case_id"].astype(str) == str(args.case_id)]
    if row.empty:
        raise SystemExit(f"case_id not found: {args.case_id}")
    r = row.iloc[0]
    flair_path = str(r.get("flair_pre_path", "") or r.get("flair_path", ""))
    mask_path = str(r.get("wmh_path", "") or r.get("mask_path", ""))
    if not flair_path or not mask_path:
        raise SystemExit("Selected case is missing flair_pre_path/flair_path or wmh_path/mask_path.")

    flair = _load_volume(flair_path, nib=nib)
    mask = _load_volume(mask_path, nib=nib)
    fg = wmh_foreground_mask(mask)
    z = _best_slice(fg, flair.shape)

    pred_fg = None
    if args.prediction:
        pred_path = Path(args.prediction)
        if not pred_path.exists():
            raise SystemExit(f"prediction not found: {pred_path}")
        pred = _load_volume(str(pred_path), nib=nib)
        pred_fg = pred.astype(bool)
        if pred_fg.shape != fg.shape:
            raise SystemExit(f"prediction shape {pred_fg.shape} != label shape {fg.shape}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pred_fg is None:
        out_path = out_dir / f"{args.case_id}_flair_label1_overlay.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(flair[:, :, z].T, cmap="gray", origin="lower")
        plt.imshow(np.ma.masked_where(~fg[:, :, z].T, fg[:, :, z].T), alpha=0.4, cmap="Greens", origin="lower")
        plt.title(f"{args.case_id} FLAIR + label==1 overlay z={z}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        return

    out_path = out_dir / f"{args.case_id}_pred_vs_gt.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title(f"{args.case_id} FLAIR z={z}")
    axes[0].axis("off")
    axes[1].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(~fg[:, :, z].T, fg[:, :, z].T), alpha=0.45, cmap="Greens", origin="lower")
    axes[1].set_title("Ground truth (label==1)")
    axes[1].axis("off")
    axes[2].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(~pred_fg[:, :, z].T, pred_fg[:, :, z].T), alpha=0.45, cmap="Reds", origin="lower")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    fig.suptitle("Local PoC visualization only; not clinical or official benchmark evidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
