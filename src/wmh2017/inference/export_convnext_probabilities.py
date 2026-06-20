"""Export 3D probability volumes from a trained WMH2017 2.5D ConvNeXt checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wmh2017.data.wmh_slice_dataset import WmhVolumeDataset
from wmh2017.inference.export_probabilities import save_case_probability_map
from wmh2017.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from wmh2017.training.mps_compat import enable_mps_cpu_fallback, resolve_training_device


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def infer_volume_probabilities(
    *,
    model: torch.nn.Module,
    image3d: np.ndarray,
    offsets: list[int],
    device: torch.device,
    img_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Slice-wise 2.5D inference; returns foreground probability (Z, H, W) at native resolution."""
    from wmh2017.data.wmh_slice_dataset import _center_pad_crop_2d_np, _restore_pad_crop_2d_np

    zdim = int(image3d.shape[0])
    orig_h, orig_w = int(image3d.shape[1]), int(image3d.shape[2])
    out_h, out_w = img_size if img_size is not None else (orig_h, orig_w)
    prob_work = np.zeros((zdim, out_h, out_w), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for z in range(zdim):
            slices = []
            for offset in offsets:
                zi = int(np.clip(z + offset, 0, zdim - 1))
                sl = image3d[zi]
                if img_size is not None:
                    sl = _center_pad_crop_2d_np(sl, out_h, out_w)
                slices.append(sl)
            inp = np.stack(slices, axis=0).astype(np.float32)
            tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
            logits = model(tensor)
            if isinstance(logits, list):
                logits = logits[0]
            prob_work[z] = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    if img_size is not None and (orig_h, orig_w) != (out_h, out_w):
        return _restore_pad_crop_2d_np(prob_work, orig_h, orig_w, out_h, out_w)
    return prob_work


def export_convnext_val_probabilities(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    probs_dir: str | Path | None = None,
    assigned_split: str = "val",
) -> dict[str, Any]:
    """Export val 3D probability maps from a ConvNeXt 2.5D checkpoint."""
    enable_mps_cpu_fallback()
    cfg = _load_config(config_path)
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    run_id = str(run_cfg.get("run_id", "wmh2017_convnext_export"))
    device, _runtime = resolve_training_device(torch, str(run_cfg.get("device", "auto")))

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    target_probs_dir = Path(probs_dir) if probs_dir is not None else out_dir / "predictions" / "probs"
    target_probs_dir.mkdir(parents=True, exist_ok=True)

    k = int(data_cfg.get("slice_k", 2))
    offsets = list(range(-k, k + 1))
    img_size_cfg = data_cfg.get("img_size")
    out_h, out_w = (int(img_size_cfg[0]), int(img_size_cfg[1])) if img_size_cfg else (None, None)
    in_channels = int(model_cfg.get("in_channels", 2 * k + 1))
    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        pretrained=False,
        out_channels=1,
    ).to(device)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)  # nosec B614
    model.load_state_dict(state["model_state_dict"])

    vol_ds = WmhVolumeDataset(
        data_cfg["dataset_manifest"],
        data_cfg["split_manifest"],
        assigned_split,
        image_key=str(data_cfg.get("image_key", "flair_pre_path")),
        label_key=str(data_cfg.get("label_key", "wmh_path")),
    )

    exported: list[str] = []
    for idx in range(len(vol_ds)):
        sample = vol_ds[idx]
        image3d = np.asarray(sample["image"], dtype=np.float32)
        probs = infer_volume_probabilities(
            model=model,
            image3d=image3d,
            offsets=offsets,
            device=device,
            img_size=(out_h, out_w) if out_h is not None and out_w is not None else None,
        )
        prob_path = target_probs_dir / f"{sample['case_id']}.npz"
        save_case_probability_map(probs, prob_path)
        exported.append(str(prob_path))

    return {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "probs_dir": str(target_probs_dir),
        "n_cases": len(exported),
        "assigned_split": assigned_split,
    }
