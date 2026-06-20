"""Train WMH2017 2.5D ConvNeXt-Tiny segmentation (active port)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.data.wmh_slice_dataset import WmhSliceDataset, WmhVolumeDataset
from wmh2017.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from wmh2017.training.losses import TverskyFocalLoss
from wmh2017.training.mps_compat import enable_mps_cpu_fallback, resolve_training_device


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_convnext_25d(config_path: str | Path) -> dict[str, Any]:
    enable_mps_cpu_fallback()
    cfg = _load_config(config_path)
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    run_id = str(run_cfg.get("run_id", "wmh2017_convnext25d"))
    seed = int(run_cfg.get("seed", 42))
    _set_seed(seed)
    device, device_runtime = resolve_training_device(torch, str(run_cfg.get("device", "auto")))

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    vol_ds = WmhVolumeDataset(
        data_cfg["dataset_manifest"],
        data_cfg["split_manifest"],
        "train",
        image_key=str(data_cfg.get("image_key", "flair_pre_path")),
        label_key=str(data_cfg.get("label_key", "wmh_path")),
    )
    val_vol_ds = WmhVolumeDataset(
        data_cfg["dataset_manifest"],
        data_cfg["split_manifest"],
        "val",
        image_key=str(data_cfg.get("image_key", "flair_pre_path")),
        label_key=str(data_cfg.get("label_key", "wmh_path")),
    )
    k = int(data_cfg.get("slice_k", 2))
    img_size_cfg = data_cfg.get("img_size")
    img_size = (int(img_size_cfg[0]), int(img_size_cfg[1])) if img_size_cfg else None
    train_ds = WmhSliceDataset(vol_ds, k=k, img_size=img_size)
    val_ds = WmhSliceDataset(val_vol_ds, k=k, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 4)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    in_channels = int(model_cfg.get("in_channels", 2 * k + 1))
    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        pretrained=bool(model_cfg.get("pretrained", True)),
        out_channels=1,
    ).to(device)
    loss_fn = TverskyFocalLoss(
        alpha=float(train_cfg.get("loss", {}).get("alpha", 0.3)),
        beta=float(train_cfg.get("loss", {}).get("beta", 0.7)),
        gamma=float(train_cfg.get("loss", {}).get("gamma", 1.33)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))

    max_epochs = int(train_cfg.get("max_epochs", 20))
    logs: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = ckpt_dir / "model_best.pt"

    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            if isinstance(logits, list):
                logits = logits[0]
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}")
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                val_losses.append(float(loss_fn(logits, labels).detach().cpu().item()))
        mean_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        logs.append({"epoch": epoch, "train_loss": mean_train, "val_loss": mean_val})
        if mean_val < best_val:
            best_val = mean_val
            torch.save(  # nosec B614
                {"run_id": run_id, "model_state_dict": model.state_dict(), "config": cfg},
                best_path,
            )

    log_path = log_dir / "train_log.jsonl"
    log_path.write_text("\n".join(json.dumps(x) for x in logs) + "\n", encoding="utf-8")
    evidence = {
        "run_id": run_id,
        "status": "completed",
        "training_mode": "convnext_25d",
        "device": str(device),
        **device_runtime,
        "checkpoint_path": str(best_path),
        "train_log": str(log_path),
        "safety": {
            "test_split_used": False,
            "label_policy": "label==1 foreground; label==2 ignored as foreground",
            "claim_boundary": "local PoC ConvNeXt 2.5D training only",
        },
    }
    write_json(out_dir / "run_evidence.json", evidence)
    append_run_manifest(
        make_run_row(
            run_id=run_id,
            run_purpose="wmh2017_convnext_25d_training",
            config_path=str(config_path),
            dataset_manifest=str(data_cfg["dataset_manifest"]),
            split_manifest=str(data_cfg["split_manifest"]),
            model_name="convnext_nnunet_25d",
            model_version="full",
            seed=seed,
            device=str(device),
            status="completed",
            checkpoint_path=str(best_path),
            prediction_dir="",
            notes=f"convnext_25d; best_val_loss_proxy={best_val:.6f}",
        ),
        run_cfg.get("run_manifest", "registry/runs/run_manifest.csv"),
    )
    return {"run_id": run_id, "checkpoint_path": str(best_path), "best_val_loss_proxy": best_val}
