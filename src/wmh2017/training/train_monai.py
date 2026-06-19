"""Executable MONAI smoke training for WMH2017.

The goal is not SOTA performance. The goal is to prove that data loading,
label==1 policy, train/val split, one short training loop, validation inference,
prediction persistence and run evidence work end-to-end.

Safety boundaries:
- challenge_split=test is never consumed by this module.
- label value 2 is ignored as foreground by converting labels to label==1.
- raw data and checkpoints remain outside git-controlled source paths by default.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh2017.audit.run_record import append_run_manifest, make_run_row, write_json
from wmh2017.data.preprocessing import normalize_nonzero_channelwise
from wmh2017.io.images import load_array, save_array_like
from wmh2017.training.mps_compat import (
    apply_mps_safe_convtranspose_patch,
    enable_mps_cpu_fallback,
    record_mps_convtranspose_patch,
    resolve_training_device,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for YAML configs. Install requirements-lock.txt.") from e
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _require_monai_stack() -> tuple[Any, Any]:
    try:
        import torch
        from monai.data import DataLoader, Dataset, decollate_batch
        from monai.inferers import sliding_window_inference
        from monai.losses import DiceCELoss
        from monai.networks.nets import UNet
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            Lambdad,
            LoadImaged,
            NormalizeIntensityd,
            RandCropByPosNegLabeld,
            ResizeWithPadOrCropd,
        )
    except ImportError as e:
        raise ImportError(
            "MONAI smoke training requires torch, monai and nibabel. "
            "Install with: python -m pip install -r requirements-lock.txt"
        ) from e
    return torch, {
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "decollate_batch": decollate_batch,
        "sliding_window_inference": sliding_window_inference,
        "DiceCELoss": DiceCELoss,
        "UNet": UNet,
        "Compose": Compose,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "EnsureTyped": EnsureTyped,
        "Lambdad": Lambdad,
        "LoadImaged": LoadImaged,
        "NormalizeIntensityd": NormalizeIntensityd,
        "RandCropByPosNegLabeld": RandCropByPosNegLabeld,
        "ResizeWithPadOrCropd": ResizeWithPadOrCropd,
    }


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _case_rows(
    manifest_csv: str | Path, split_csv: str | Path, assigned_split: str, image_key: str, label_key: str
) -> list[dict[str, str]]:
    manifest = pd.read_csv(manifest_csv)
    split = pd.read_csv(split_csv)
    split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].copy()
    if split.empty:
        raise ValueError(f"no rows for assigned_split={assigned_split} in {split_csv}")
    rows: list[dict[str, str]] = []
    for _, s in split.iterrows():
        case_id = str(s["case_id"])
        m = manifest[manifest["case_id"].astype(str) == case_id]
        if m.empty:
            raise ValueError(f"case_id={case_id} exists in split but not manifest")
        r = m.iloc[0]
        if str(r.get("challenge_split", "")).lower() == "test":
            raise ValueError(f"test case cannot be used for train/val smoke training: {case_id}")
        image = str(r.get(image_key, "") or r.get("flair_path", "") or r.get("flair_pre_path", ""))
        label = str(r.get(label_key, "") or r.get("wmh_path", "") or r.get("mask_path", ""))
        if not image or not label:
            raise ValueError(f"case_id={case_id} missing image or label path")
        rows.append({"case_id": case_id, "image": image, "label": label})
    return rows


def _build_model(monai: dict[str, Any], cfg: dict[str, Any]) -> Any:
    m = cfg.get("model", {})
    return monai["UNet"](
        spatial_dims=int(m.get("spatial_dims", 3)),
        in_channels=int(m.get("in_channels", 1)),
        out_channels=int(m.get("out_channels", 2)),
        channels=tuple(m.get("channels", [8, 16, 32])),
        strides=tuple(m.get("strides", [2, 2])),
        num_res_units=int(m.get("num_res_units", 1)),
    )


def _transforms(monai: dict[str, Any], patch_size: list[int] | tuple[int, int, int], train: bool) -> Any:
    ops = [
        monai["LoadImaged"](keys=["image", "label"]),
        monai["EnsureChannelFirstd"](keys=["image", "label"]),
        monai["Lambdad"](keys=["image"], func=normalize_nonzero_channelwise),
        monai["Lambdad"](keys=["label"], func=lambda x: (x == 1).astype(np.int64)),
    ]
    if train:
        ops.append(
            monai["RandCropByPosNegLabeld"](
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(patch_size),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            )
        )
    ops.extend(
        [
            monai["ResizeWithPadOrCropd"](keys=["image", "label"], spatial_size=tuple(patch_size)),
            monai["EnsureTyped"](keys=["image", "label"]),
        ]
    )
    return monai["Compose"](ops)


def _normalize_for_inference(image: np.ndarray) -> np.ndarray:
    return normalize_nonzero_channelwise(image)


def main(config_path: str) -> None:
    enable_mps_cpu_fallback()
    torch, monai = _require_monai_stack()
    cfg = _load_config(config_path)

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    run_id = str(run_cfg.get("run_id", "wmh2017_monai_smoke"))
    seed = int(run_cfg.get("seed", 42))
    _set_seed(seed, torch)
    device_requested = str(run_cfg.get("device", "auto"))
    device, device_runtime = resolve_training_device(torch, device_requested)

    dataset_manifest = str(data_cfg["dataset_manifest"])
    split_manifest = str(data_cfg["split_manifest"])
    image_key = str(data_cfg.get("image_key", "flair_pre_path"))
    label_key = str(data_cfg.get("label_key", "wmh_path"))
    patch_size = list(data_cfg.get("patch_size", [32, 32, 32]))
    num_workers = int(data_cfg.get("num_workers", 0))
    val_max_cases = int(data_cfg.get("val_max_cases", 2))

    out_dir = Path(run_cfg.get("output_dir", f"artifacts/runs/{run_id}"))
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    for p in [pred_dir, ckpt_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    train_rows = _case_rows(dataset_manifest, split_manifest, "train", image_key, label_key)
    val_rows = _case_rows(dataset_manifest, split_manifest, "val", image_key, label_key)[:val_max_cases]
    if not train_rows or not val_rows:
        raise ValueError("train and val rows are required for smoke training")

    train_ds = monai["Dataset"](data=train_rows, transform=_transforms(monai, patch_size, train=True))
    train_loader = monai["DataLoader"](train_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    model = _build_model(monai, cfg)
    if device.type == "mps":
        patched_layers = apply_mps_safe_convtranspose_patch(model)
        if patched_layers == 0:
            raise RuntimeError(
                "MPS was selected but no ConvTranspose3d layers were replaced; "
                "MONAI smoke cannot run on MPS without the compatibility patch."
            )
        device_runtime = record_mps_convtranspose_patch(device_runtime, patched_layers)
    model = model.to(device)
    loss_fn = monai["DiceCELoss"](to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))

    max_epochs = int(train_cfg.get("max_epochs", 1))
    max_steps = int(train_cfg.get("max_steps_per_epoch", 2))
    logs: list[dict[str, Any]] = []

    model.train()
    global_step = 0
    for epoch in range(max_epochs):
        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}: {loss.item()}")
            loss.backward()
            opt.step()
            global_step += 1
            logs.append(
                {"epoch": epoch, "step": step, "global_step": global_step, "loss": float(loss.detach().cpu().item())}
            )

    checkpoint_path = ""
    if bool(train_cfg.get("save_checkpoint", True)):
        checkpoint_path = str(ckpt_dir / "model_smoke.pt")
        torch.save(  # nosec B614 — local smoke checkpoint only; not loading untrusted weights
            {
                "run_id": run_id,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "global_step": global_step,
                "claim_boundary": "smoke checkpoint only; not a clinical or production model",
            },
            checkpoint_path,
        )

    # Validation inference on original volume. Sliding-window keeps memory bounded.
    if bool(train_cfg.get("save_predictions", True)):
        model.eval()
        roi_size = tuple(patch_size)
        sw_batch_size = 1
        with torch.no_grad():
            for row in val_rows:
                image = load_array(row["image"])
                x = _normalize_for_inference(image)
                tensor = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
                logits = monai["sliding_window_inference"](
                    tensor, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model
                )
                probs = torch.softmax(logits, dim=1)[:, 1]
                pred = (probs[0].detach().cpu().numpy() >= float(train_cfg.get("threshold", 0.5))).astype(np.uint8)
                save_array_like(row["image"], pred_dir / f"{row['case_id']}_pred.nii.gz", pred)

    log_path = log_dir / "train_log.jsonl"
    log_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in logs) + "\n", encoding="utf-8")

    evidence = {
        "run_id": run_id,
        "status": "completed",
        "device": str(device),
        **device_runtime,
        "global_step": global_step,
        "train_case_count": len(train_rows),
        "val_prediction_count": len(list(pred_dir.glob("*_pred.*"))),
        "train_log": str(log_path),
        "checkpoint_path": checkpoint_path,
        "prediction_dir": str(pred_dir),
        "safety": {
            "test_split_used": False,
            "label_policy": "label==1 foreground; label==2 ignored as foreground",
            "claim_boundary": "local PoC smoke only",
            "mps_execution_claim": (
                "MPS-compatible patched smoke; not native-MPS equivalence with ConvTranspose3d"
                if device.type == "mps"
                else "standard device path without MPS ConvTranspose3d patch"
            ),
        },
    }
    evidence_path = out_dir / "run_evidence.json"
    write_json(evidence_path, evidence)

    row = make_run_row(
        run_id=run_id,
        run_purpose="wmh2017_monai_smoke_training",
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        model_name=str(cfg.get("model", {}).get("name", "monai_unet")),
        model_version="smoke",
        seed=seed,
        device=str(device),
        status="completed",
        checkpoint_path=checkpoint_path,
        prediction_dir=str(pred_dir),
        notes=f"global_step={global_step}; smoke only; no test110 use",
    )
    append_run_manifest(row, run_cfg.get("run_manifest", "registry/runs/run_manifest.csv"))

    print(f"Completed MONAI smoke training run_id={run_id}")
    print(f"Wrote train log: {log_path}")
    print(f"Wrote run evidence: {evidence_path}")
    if checkpoint_path:
        print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote predictions: {pred_dir}")
