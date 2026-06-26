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


class SmallLesionWeightedDiceCELoss(nn.Module):
    """Dice + small-lesion-weighted cross-entropy (default-off lever; CE weighting only).

    Dice term is the standard soft-dice on foreground probability (unchanged region term). The CE
    term is per-voxel cross-entropy re-weighted by a small-lesion map derived from the GT:
    ``weighted_ce = sum(ce * w) / sum(w)`` with ``w = small_lesion_ce_weight`` on voxels in GT
    connected components of size <= ``small_lesion_max_voxels`` and 1 elsewhere. With weight 1.0 the
    map is all ones and this reduces to dice + mean CE.
    """

    def __init__(
        self,
        small_lesion_ce_weight: float,
        small_lesion_max_voxels: int = 10,
        connectivity: int = 26,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.small_lesion_ce_weight = float(small_lesion_ce_weight)
        self.small_lesion_max_voxels = int(small_lesion_max_voxels)
        self.connectivity = int(connectivity)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import numpy as np

        from wmh2017.training.small_lesion_sampling import small_lesion_weight_map

        # target_idx: (B, *spatial) long
        target_idx = targets[:, 0].long() if targets.ndim == logits.ndim else targets.long()
        targets_f = target_idx.unsqueeze(1).float()

        # soft dice on foreground probability (region term, unchanged)
        probs = _foreground_probs_from_logits(logits)
        probs_flat, targets_flat = _flatten_spatial(probs, targets_f)
        inter = (probs_flat * targets_flat).sum(1)
        den = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()

        # per-voxel CE, re-weighted by the small-lesion map (normalized)
        ce_voxel = F.cross_entropy(logits, target_idx, reduction="none")
        ti = target_idx.detach().cpu().numpy()
        wmaps = np.stack(
            [
                small_lesion_weight_map(
                    ti[b],
                    self.small_lesion_max_voxels,
                    self.small_lesion_ce_weight,
                    connectivity=self.connectivity,
                )
                for b in range(ti.shape[0])
            ]
        )
        wmap = torch.as_tensor(wmaps, dtype=ce_voxel.dtype, device=ce_voxel.device)
        weighted_ce = (ce_voxel * wmap).sum() / wmap.sum().clamp_min(1.0)
        return dice_loss + weighted_ce


class MonaiDiceCELossWrapper(nn.Module):
    """Adapter around MONAI DiceCELoss for config-driven factory parity."""

    def __init__(self, monai_loss: Any) -> None:
        super().__init__()
        self.monai_loss = monai_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.monai_loss(logits, targets)
