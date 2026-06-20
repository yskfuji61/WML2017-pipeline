"""WMH2017 2.5D slice dataset (active port; label==1 foreground only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.io.images import load_array, load_image_metadata


def _center_pad_crop_2d_np(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Center-pad or center-crop a 2D array to (out_h, out_w)."""
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    if pad_h or pad_w:
        pad_top, pad_left = pad_h // 2, pad_w // 2
        arr = np.pad(
            arr,
            ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
            mode="constant",
            constant_values=0.0,
        )
    h2, w2 = int(arr.shape[-2]), int(arr.shape[-1])
    if h2 > out_h:
        top = (h2 - out_h) // 2
        arr = arr[top : top + out_h, :]
    if w2 > out_w:
        left = (w2 - out_w) // 2
        arr = arr[:, left : left + out_w]
    return arr.astype(np.float32, copy=False)


def _restore_pad_crop_2d_np(pred: np.ndarray, orig_h: int, orig_w: int, out_h: int, out_w: int) -> np.ndarray:
    """Inverse of _center_pad_crop_2d_np for a (Z, out_h, out_w) or (out_h, out_w) map."""
    arr = np.asarray(pred, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if orig_h > out_h:
        t = (orig_h - out_h) // 2
        arr = np.pad(arr, ((0, 0), (t, orig_h - out_h - t), (0, 0)), constant_values=0.0)
    if orig_w > out_w:
        left = (orig_w - out_w) // 2
        arr = np.pad(arr, ((0, 0), (0, 0), (left, orig_w - out_w - left)), constant_values=0.0)
    h_cur, w_cur = int(arr.shape[-2]), int(arr.shape[-1])
    if h_cur > orig_h:
        top = (h_cur - orig_h) // 2
        arr = arr[:, top : top + orig_h, :]
    if w_cur > orig_w:
        left = (w_cur - orig_w) // 2
        arr = arr[:, :, left : left + orig_w]
    return arr[0] if squeeze else arr


class WmhVolumeDataset(Dataset):
    """Load full FLAIR volumes with WMH label==1 foreground masks."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split_csv: str | Path,
        assigned_split: str,
        *,
        image_key: str = "flair_pre_path",
        label_key: str = "wmh_path",
    ) -> None:
        manifest = pd.read_csv(manifest_csv)
        split = pd.read_csv(split_csv)
        split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].reset_index(drop=True)
        if split.empty:
            raise ValueError(f"no rows for assigned_split={assigned_split}")
        self.rows: list[dict[str, str]] = []
        for _, srow in split.iterrows():
            case_id = str(srow["case_id"])
            mrow = manifest[manifest["case_id"].astype(str) == case_id]
            if mrow.empty:
                raise ValueError(f"case_id={case_id} missing in manifest")
            m = mrow.iloc[0]
            if str(m.get("challenge_split", "")).lower() == "test":
                raise ValueError(f"test case {case_id} cannot be used for training/val")
            image = str(m.get(image_key, "") or m.get("flair_pre_path", ""))
            label = str(m.get(label_key, "") or m.get("wmh_path", ""))
            if not image or not label:
                raise ValueError(f"case_id={case_id} missing image/label path")
            self.rows.append({"case_id": case_id, "image": image, "label": label})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        image = load_array(row["image"]).astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        label = wmh_foreground_mask(load_array(row["label"])).astype(np.float32)
        return {"case_id": row["case_id"], "image": image, "label": label}


class WmhSliceDataset(Dataset):
    """2.5D slices stacked along channel dimension for ConvNeXt training."""

    def __init__(
        self,
        volume_dataset: WmhVolumeDataset,
        *,
        k: int = 2,
        slice_offsets: list[int] | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> None:
        self.volume_dataset = volume_dataset
        self.k = int(k)
        self.offsets = list(slice_offsets) if slice_offsets is not None else list(range(-k, k + 1))
        self.img_size = tuple(img_size) if img_size is not None else None
        self.index_map: list[tuple[int, int]] = []
        for vidx in range(len(volume_dataset)):
            row = volume_dataset.rows[vidx]
            meta = load_image_metadata(row["image"])
            shape = meta.shape
            zdim = int(shape[0]) if len(shape) == 3 else int(shape[1])
            for z in range(zdim):
                self.index_map.append((vidx, z))
        self._cache_vidx: int | None = None
        self._cache: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        vidx, z = self.index_map[int(idx)]
        if self._cache_vidx != vidx or self._cache is None:
            self._cache = self.volume_dataset[int(vidx)]
            self._cache_vidx = vidx
        image3d = self._cache["image"]
        label3d = self._cache["label"]
        slices = []
        for offset in self.offsets:
            zi = int(np.clip(z + offset, 0, image3d.shape[0] - 1))
            slices.append(image3d[zi])
        image2_5d = np.stack(slices, axis=0).astype(np.float32)
        mask2d = label3d[z].astype(np.float32)
        if self.img_size is not None:
            out_h, out_w = self.img_size
            image2_5d = np.stack([_center_pad_crop_2d_np(s, out_h, out_w) for s in image2_5d], axis=0)
            mask2d = _center_pad_crop_2d_np(mask2d, out_h, out_w)
        return {
            "image": torch.from_numpy(image2_5d),
            "label": torch.from_numpy(mask2d[None, ...]),
            "case_id": self._cache["case_id"],
            "z": z,
        }
