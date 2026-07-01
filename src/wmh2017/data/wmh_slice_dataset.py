"""WMH2017 2.5D slice dataset (active port; label==1 foreground only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wmh2017.data.label_policy import wmh_foreground_mask
from wmh2017.data.split_guard import guard_challenge_split_test
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


# T1-R8: default-off FLAIR+T1 multimodal support. Single-modality path is bit-identical.
MODALITY_KEY_MAP = {"flair": "flair_pre_path", "t1": "t1_pre_path"}


def resolve_modality_keys(modalities: list[str]) -> list[str]:
    """Map modality names to manifest keys; unknown name raises a clear ValueError."""
    keys: list[str] = []
    for name in modalities:
        n = str(name).strip().lower()
        if n not in MODALITY_KEY_MAP:
            raise ValueError(f"unsupported modality {name!r}; expected one of {sorted(MODALITY_KEY_MAP)}")
        keys.append(MODALITY_KEY_MAP[n])
    return keys


def stack_slices_2_5d(
    volumes: list[np.ndarray],
    z: int,
    offsets: list[int],
    img_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Stack 2.5D slices modality-major: for each volume its 2k+1 offset slices, blocks concatenated.

    Channel order = block per modality in the given order (e.g. FLAIR block then T1 block). A single
    volume yields ``(len(offsets), H, W)`` identical to the prior single-modality behavior.
    """
    blocks: list[np.ndarray] = []
    for vol in volumes:
        zmax = int(vol.shape[0]) - 1
        for offset in offsets:
            zi = int(np.clip(int(z) + int(offset), 0, zmax))
            sl = vol[zi]
            if img_size is not None:
                sl = _center_pad_crop_2d_np(sl, int(img_size[0]), int(img_size[1]))
            blocks.append(np.asarray(sl, dtype=np.float32))
    return np.stack(blocks, axis=0).astype(np.float32)


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
        image_keys: list[str] | None = None,
    ) -> None:
        manifest = pd.read_csv(manifest_csv)
        split = pd.read_csv(split_csv)
        split = split[split["assigned_split"].astype(str).str.lower() == assigned_split.lower()].reset_index(drop=True)
        if split.empty:
            raise ValueError(f"no rows for assigned_split={assigned_split}")
        # Default-off multimodal: image_keys is None -> single-modality (bit-identical).
        self.multimodal = image_keys is not None
        self.rows: list[dict[str, Any]] = []
        for _, srow in split.iterrows():
            case_id = str(srow["case_id"])
            mrow = manifest[manifest["case_id"].astype(str) == case_id]
            if mrow.empty:
                raise ValueError(f"case_id={case_id} missing in manifest")
            m = mrow.iloc[0]
            guard_challenge_split_test(
                case_id, str(m.get("challenge_split", "")), assigned_split=assigned_split, context="slice_dataset"
            )
            label = str(m.get(label_key, "") or m.get("wmh_path", ""))
            if not label:
                raise ValueError(f"case_id={case_id} missing label path")
            if self.multimodal:
                paths: list[str] = []
                for key in image_keys:  # type: ignore[union-attr]
                    raw = m.get(key, "")
                    p = "" if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else str(raw).strip()
                    if not p:
                        raise ValueError(f"case_id={case_id} missing modality path for key={key!r}")
                    paths.append(p)
                # row["image"] = first modality (used for slice-count metadata); image_paths = all.
                self.rows.append({"case_id": case_id, "image": paths[0], "image_paths": paths, "label": label})
            else:
                image = str(m.get(image_key, "") or m.get("flair_pre_path", ""))
                if not image:
                    raise ValueError(f"case_id={case_id} missing image/label path")
                self.rows.append({"case_id": case_id, "image": image, "label": label})

    def __len__(self) -> int:
        return len(self.rows)

    def _load_volume(self, path: str) -> np.ndarray:
        vol = load_array(path).astype(np.float32)
        if vol.ndim == 4:
            vol = vol[0]
        return vol

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        label = wmh_foreground_mask(load_array(row["label"])).astype(np.float32)
        if self.multimodal:
            images = [self._load_volume(p) for p in row["image_paths"]]
            return {"case_id": row["case_id"], "image": images[0], "images": images, "label": label}
        image = self._load_volume(row["image"])
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
        label3d = self._cache["label"]
        # Multimodal (T1-R8) stacks FLAIR block then T1 block; single-modality is unchanged.
        volumes = self._cache["images"] if "images" in self._cache else [self._cache["image"]]
        image2_5d = stack_slices_2_5d(volumes, z, self.offsets, self.img_size)
        mask2d = label3d[z].astype(np.float32)
        if self.img_size is not None:
            out_h, out_w = self.img_size
            mask2d = _center_pad_crop_2d_np(mask2d, out_h, out_w)
        return {
            "image": torch.from_numpy(image2_5d),
            "label": torch.from_numpy(mask2d[None, ...]),
            "case_id": self._cache["case_id"],
            "z": z,
        }
