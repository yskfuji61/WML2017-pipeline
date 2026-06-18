"""Small image IO layer.

NIfTI is used for real WMH2017 runs. NumPy .npy is supported for CI fixtures so
integration tests do not need to vendor medical images.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    """Minimal geometry contract for local WMH2017 evaluation.

    `affine_sha256` is empty for array formats that do not carry affine metadata.
    `spacing` is empty when spacing is unavailable. The evaluation layer treats
    NIfTI geometry as a hard contract and `.npy` fixtures as shape-only.
    """

    path: str
    format: str
    shape: tuple[int, ...]
    dtype: str
    spacing: tuple[float, ...]
    affine_sha256: str

    def to_record(self, prefix: str) -> dict[str, Any]:
        data = asdict(self)
        return {
            f"{prefix}_shape": "x".join(str(x) for x in data["shape"]),
            f"{prefix}_dtype": data["dtype"],
            f"{prefix}_spacing": "x".join(f"{float(x):.8g}" for x in data["spacing"]),
            f"{prefix}_affine_sha256": data["affine_sha256"],
            f"{prefix}_format": data["format"],
        }


def _format_for_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".npy"):
        return "npy"
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "nifti"
    raise ValueError(f"unsupported image format for {path}; expected .npy, .nii or .nii.gz")


def _require_existing_supported_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image path not found: {p}")
    _format_for_path(p)
    return p


def _sha256_array_bytes(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def load_array(path: str | Path) -> np.ndarray:
    p = _require_existing_supported_path(path)
    fmt = _format_for_path(p)
    if fmt == "npy":
        return np.load(p)
    if fmt == "nifti":
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError("nibabel is required to read NIfTI files. Install requirements-lock.txt.") from e
        return np.asarray(nib.load(str(p)).get_fdata())
    raise AssertionError(f"unreachable image format branch: {fmt}")


def load_image_metadata(path: str | Path) -> ImageMetadata:
    """Load shape/spacing/affine metadata without copying raw image data into the repo."""
    p = _require_existing_supported_path(path)
    fmt = _format_for_path(p)

    if fmt == "npy":
        arr = np.load(p, mmap_mode="r")
        return ImageMetadata(
            path=str(p),
            format=fmt,
            shape=tuple(int(x) for x in arr.shape),
            dtype=str(arr.dtype),
            spacing=(),
            affine_sha256="",
        )

    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required to inspect NIfTI metadata. Install requirements-lock.txt.") from e

    img = nib.load(str(p))
    affine = np.asarray(img.affine, dtype=np.float64)
    return ImageMetadata(
        path=str(p),
        format=fmt,
        shape=tuple(int(x) for x in img.shape),
        dtype=str(img.get_data_dtype()),
        spacing=tuple(float(x) for x in img.header.get_zooms()[: len(img.shape)]),
        affine_sha256=_sha256_array_bytes(affine),
    )


def assert_compatible_image_geometry(
    prediction: ImageMetadata,
    label: ImageMetadata,
    *,
    case_id: str,
    require_affine_match: bool = True,
    require_spacing_match: bool = True,
) -> None:
    """Fail closed when prediction/label arrays cannot be safely compared.

    Shape mismatch is always fatal. For NIfTI-vs-NIfTI comparison, spacing and
    affine are also fatal by default because voxel-wise metrics and HD95/AVD
    can become invalid or misleading when geometry differs.
    """
    if prediction.shape != label.shape:
        raise ValueError(
            f"prediction/label shape mismatch for case_id={case_id}: "
            f"prediction_shape={prediction.shape}, label_shape={label.shape}"
        )

    both_nifti = prediction.format == "nifti" and label.format == "nifti"
    if both_nifti and require_spacing_match and prediction.spacing != label.spacing:
        raise ValueError(
            f"prediction/label spacing mismatch for case_id={case_id}: "
            f"prediction_spacing={prediction.spacing}, label_spacing={label.spacing}"
        )

    if both_nifti and require_affine_match and prediction.affine_sha256 != label.affine_sha256:
        raise ValueError(
            f"prediction/label affine mismatch for case_id={case_id}: "
            f"prediction_affine_sha256={prediction.affine_sha256}, "
            f"label_affine_sha256={label.affine_sha256}"
        )


def save_array_like(reference_path: str | Path, output_path: str | Path, array: np.ndarray) -> None:
    """Save prediction using NIfTI affine/header when possible; otherwise .npy."""
    ref = Path(reference_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_format = _format_for_path(out)
    if out_format == "npy":
        np.save(out, array)
        return
    if out_format == "nifti":
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError("nibabel is required to write NIfTI files. Install requirements-lock.txt.") from e
        img = nib.load(str(ref))
        pred = np.asarray(array).astype(np.uint8)
        nib.save(nib.Nifti1Image(pred, img.affine, img.header), str(out))
        return
    raise AssertionError(f"unreachable image format branch: {out_format}")


def image_shape(path: str | Path) -> tuple[int, ...]:
    return load_image_metadata(path).shape
