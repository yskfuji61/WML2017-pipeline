"""Config-driven MONAI transform factory for WMH2017 training."""

from __future__ import annotations

from typing import Any

import numpy as np

from wmh2017.data.preprocessing import normalize_nonzero_channelwise
from wmh2017.training.small_lesion_sampling import (
    SmallLesionFgBgIndicesd,
    resolve_small_lesion_sampling_cfg,
)


def _label_to_foreground_mask(label: np.ndarray) -> np.ndarray:
    return (label == 1).astype(np.int64)


class StackModalitiesd:
    """Concatenate channel-first modality arrays into a single ``output_key`` volume.

    Each input key is expected to be channel-first (e.g. ``(1, Z, Y, X)`` after
    ``EnsureChannelFirstd``); the result is stacked along the channel axis. Keys other
    than ``output_key`` are dropped after stacking. Only used for multi-modality inputs.
    """

    def __init__(self, keys: tuple[str, ...], output_key: str = "image") -> None:
        self.keys = keys
        self.output_key = output_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        channels = [np.asarray(data[key]) for key in self.keys]
        data[self.output_key] = np.concatenate(channels, axis=0)
        for key in self.keys:
            if key != self.output_key:
                data.pop(key, None)
        return data


def build_monai_transforms(
    monai: dict[str, Any],
    patch_size: list[int] | tuple[int, int, int],
    *,
    train: bool,
    train_cfg: dict[str, Any] | None = None,
    input_keys: tuple[str, ...] = ("image",),
) -> Any:
    """Build MONAI Compose pipeline from training config.

    ``input_keys`` are the modality dictionary keys to load and normalize. For the
    default single ``("image",)`` the op list is identical to the prior FLAIR-only
    pipeline. With multiple keys, channels are stacked into ``"image"`` before cropping.
    """
    train_cfg = train_cfg or {}
    sampling_cfg = train_cfg.get("sampling", {}) if isinstance(train_cfg.get("sampling"), dict) else {}
    aug_cfg = train_cfg.get("augmentation", {}) if isinstance(train_cfg.get("augmentation"), dict) else {}

    load_keys = [*input_keys, "label"]
    ops: list[Any] = [
        monai["LoadImaged"](keys=load_keys),
        monai["EnsureChannelFirstd"](keys=load_keys),
        monai["Lambdad"](keys=list(input_keys), func=normalize_nonzero_channelwise),
        monai["Lambdad"](keys=["label"], func=_label_to_foreground_mask),
    ]
    if len(input_keys) > 1:
        ops.append(StackModalitiesd(keys=input_keys, output_key="image"))

    if train:
        pos = int(sampling_cfg.get("pos", 1))
        neg = int(sampling_cfg.get("neg", 1))
        num_samples = int(sampling_cfg.get("num_samples", 1))
        sl_enabled, sl_prob, sl_max_voxels = resolve_small_lesion_sampling_cfg(sampling_cfg)
        crop_kwargs: dict[str, Any] = dict(
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
        if sl_enabled:
            # Default-off small-lesion-aware sampling: precompute small-lesion-biased fg indices
            # so positive crop centers favor small lesions; everything else stays MONAI's crop.
            ops.append(
                SmallLesionFgBgIndicesd(
                    label_key="label",
                    image_key="image",
                    max_voxels=sl_max_voxels,
                    small_center_prob=sl_prob,
                    image_threshold=0,
                )
            )
            crop_kwargs["fg_indices_key"] = "fg_indices"
            crop_kwargs["bg_indices_key"] = "bg_indices"
        ops.append(monai["RandCropByPosNegLabeld"](**crop_kwargs))

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
