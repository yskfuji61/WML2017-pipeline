#!/usr/bin/env python3
"""Export validation probability maps from a trained MONAI checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmh2017.inference.export_probabilities import export_val_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Export val probability maps (.npz) from a WMH2017 MONAI checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probs-dir", default="")
    parser.add_argument("--assigned-split", default="val")
    parser.add_argument("--save-binary-predictions", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = export_val_probabilities(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        probs_dir=args.probs_dir or None,
        assigned_split=args.assigned_split,
        save_binary_predictions=args.save_binary_predictions,
        threshold=args.threshold,
    )
    print(f"Exported {result['n_cases']} probability maps to {result['probs_dir']}")


if __name__ == "__main__":
    main()
