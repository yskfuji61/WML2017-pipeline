"""Config-driven loss factory for MONAI WMH2017 training."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from wmh2017.training.losses import (
    DiceFocalLoss,
    MonaiDiceCELossWrapper,
    TverskyFocalLoss,
    TverskyLoss,
)


def build_loss(train_cfg: dict[str, Any], monai: dict[str, Any]) -> nn.Module:
    """Build training loss from config; default is MONAI DiceCELoss."""
    loss_cfg = train_cfg.get("loss")
    if not loss_cfg:
        return MonaiDiceCELossWrapper(monai["DiceCELoss"](to_onehot_y=True, softmax=True))

    if isinstance(loss_cfg, str):
        name = loss_cfg.strip().lower()
        params: dict[str, Any] = {}
    elif isinstance(loss_cfg, dict):
        name = str(loss_cfg.get("name", "dice_ce")).strip().lower()
        params = {k: v for k, v in loss_cfg.items() if k != "name"}
    else:
        raise ValueError(f"training.loss must be str or dict, got {type(loss_cfg)!r}")

    if name in {"dice_ce", "dicece", "monai_dice_ce"}:
        return MonaiDiceCELossWrapper(monai["DiceCELoss"](to_onehot_y=True, softmax=True))
    if name in {"tversky", "tversky_loss"}:
        return TverskyLoss(
            alpha=float(params.get("alpha", 0.3)),
            beta=float(params.get("beta", 0.7)),
            smooth=float(params.get("smooth", 1.0)),
        )
    if name in {"tversky_focal", "tverskyfocal"}:
        return TverskyFocalLoss(
            alpha=float(params.get("alpha", 0.3)),
            beta=float(params.get("beta", 0.7)),
            gamma=float(params.get("gamma", 1.33)),
            smooth=float(params.get("smooth", 1.0)),
        )
    if name in {"dice_focal", "dicefocal"}:
        return DiceFocalLoss(
            alpha=float(params.get("alpha", 0.25)),
            gamma=float(params.get("gamma", 2.0)),
            smooth=float(params.get("smooth", 1.0)),
        )
    raise ValueError(f"unsupported training.loss.name={name!r}")
