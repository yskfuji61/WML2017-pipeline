"""Config-driven MONAI transform factory for WMH2017 training."""

from __future__ import annotations

from typing import Any

import numpy as np

from wmh2017.data.preprocessing import normalize_nonzero_channelwise


def _label_to_foreground_mask(label: np.ndarray) -> np.ndarray:
    return (label == 1).astype(np.int64)


def build_monai_transforms(
    monai: dict[str, Any],
    patch_size: list[int] | tuple[int, int, int],
    *,
    train: bool,
    train_cfg: dict[str, Any] | None = None,
) -> Any:
    """Build MONAI Compose pipeline from training config."""
    train_cfg = train_cfg or {}
    sampling_cfg = train_cfg.get("sampling", {}) if isinstance(train_cfg.get("sampling"), dict) else {}
    aug_cfg = train_cfg.get("augmentation", {}) if isinstance(train_cfg.get("augmentation"), dict) else {}

    ops: list[Any] = [
        monai["LoadImaged"](keys=["image", "label"]),
        monai["EnsureChannelFirstd"](keys=["image", "label"]),
        monai["Lambdad"](keys=["image"], func=normalize_nonzero_channelwise),
        monai["Lambdad"](keys=["label"], func=_label_to_foreground_mask),
    ]

    if train:
        pos = int(sampling_cfg.get("pos", 1))
        neg = int(sampling_cfg.get("neg", 1))
        num_samples = int(sampling_cfg.get("num_samples", 1))
        ops.append(
            monai["RandCropByPosNegLabeld"](
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(patch_size),
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            )
        )

        flip_prob = float(aug_cfg.get("random_flip_prob", 0.0))
        if flip_prob > 0:
            ops.append(monai["RandFlipd"](keys=["image", "label"], prob=flip_prob, spatial_axis=[0, 1, 2]))

        affine_prob = float(aug_cfg.get("random_affine_prob", 0.0))
        if affine_prob > 0:
            ops.append(
                monai["RandAffined"](
                    keys=["image", "label"],
                    prob=affine_prob,
                    rotate_range=(
                        float(aug_cfg.get("rotate_range_x", 0.1)),
                        float(aug_cfg.get("rotate_range_y", 0.1)),
                        float(aug_cfg.get("rotate_range_z", 0.1)),
                    ),
                    scale_range=(
                        float(aug_cfg.get("scale_range_x", 0.1)),
                        float(aug_cfg.get("scale_range_y", 0.1)),
                        float(aug_cfg.get("scale_range_z", 0.1)),
                    ),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                )
            )

        intensity_prob = float(aug_cfg.get("random_intensity_shift_prob", 0.0))
        if intensity_prob > 0:
            ops.append(
                monai["RandShiftIntensityd"](
                    keys=["image"],
                    prob=intensity_prob,
                    offsets=float(aug_cfg.get("intensity_shift_offset", 0.1)),
                )
            )

    ops.extend(
        [
            monai["ResizeWithPadOrCropd"](keys=["image", "label"], spatial_size=tuple(patch_size)),
            monai["EnsureTyped"](keys=["image", "label"]),
        ]
    )
    return monai["Compose"](ops)
