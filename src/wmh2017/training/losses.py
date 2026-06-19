"""WMH2017 segmentation losses (active port; no legacy imports)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _foreground_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert 2-class softmax logits to foreground probability."""
    if logits.ndim < 2:
        raise ValueError(f"expected logits with channel dim, got shape={tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] >= 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise ValueError(f"unsupported logits channel count: {logits.shape[1]}")


def _flatten_spatial(probs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs_flat = torch.flatten(probs, start_dim=1)
    targets_flat = torch.flatten(targets, start_dim=1)
    return probs_flat, targets_flat


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        if targets_f.ndim == logits.ndim - 1:
            targets_f = targets_f.unsqueeze(1)
        probs = _foreground_probs_from_logits(logits)
        probs_flat, targets_flat = _flatten_spatial(probs, targets_f)

        inter = (probs_flat * targets_flat).sum(1)
        den = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()

        ce = F.binary_cross_entropy(probs, targets_f, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha * targets_f + (1.0 - self.alpha) * (1.0 - targets_f)
        focal = (alpha_t * (1 - pt) ** self.gamma * ce).mean()
        return dice_loss + focal


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        if targets_f.ndim == logits.ndim - 1:
            targets_f = targets_f.unsqueeze(1)
        probs = _foreground_probs_from_logits(logits)
        probs_flat, targets_flat = _flatten_spatial(probs, targets_f)

        tp = (probs_flat * targets_flat).sum(1)
        fp = (probs_flat * (1 - targets_flat)).sum(1)
        fn = ((1 - probs_flat) * targets_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class TverskyFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        if targets_f.ndim == logits.ndim - 1:
            targets_f = targets_f.unsqueeze(1)
        probs = _foreground_probs_from_logits(logits)
        probs_flat, targets_flat = _flatten_spatial(probs, targets_f)

        tp = (probs_flat * targets_flat).sum(1)
        fp = (probs_flat * (1 - targets_flat)).sum(1)
        fn = ((1 - probs_flat) * targets_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_t = (1 - tversky) ** self.gamma
        return focal_t.mean()


class MonaiDiceCELossWrapper(nn.Module):
    """Adapter around MONAI DiceCELoss for config-driven factory parity."""

    def __init__(self, monai_loss: Any) -> None:
        super().__init__()
        self.monai_loss = monai_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.monai_loss(logits, targets)
