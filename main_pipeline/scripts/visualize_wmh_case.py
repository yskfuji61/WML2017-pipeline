#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.data.label_policy import wmh_foreground_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a simple FLAIR + WMH label==1 overlay for one case.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True)
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
    flair_path = r.get("flair_pre_path", "") or r.get("flair_path", "")
    mask_path = r.get("wmh_path", "") or r.get("mask_path", "")
    if not flair_path or not mask_path:
        raise SystemExit("Selected case is missing flair_pre_path/flair_path or wmh_path/mask_path.")
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    fg = wmh_foreground_mask(mask)
    z = int(np.argmax(fg.sum(axis=(0, 1)))) if fg.any() else flair.shape[2] // 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_flair_label1_overlay.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(flair[:, :, z].T, cmap="gray", origin="lower")
    plt.imshow(np.ma.masked_where(~fg[:, :, z].T, fg[:, :, z].T), alpha=0.4, origin="lower")
    plt.title(f"{args.case_id} FLAIR + label==1 overlay z={z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
