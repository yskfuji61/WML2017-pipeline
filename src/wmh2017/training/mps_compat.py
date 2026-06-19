"""Apple MPS compatibility helpers for 3D segmentation models.

PyTorch exposes ``PYTORCH_ENABLE_MPS_FALLBACK=1`` for ops that lack MPS kernels,
but ``ConvTranspose3d`` still raises at runtime on MPS (PyTorch 2.4.x). The
MONAI smoke UNet therefore replaces decoder ``ConvTranspose3d`` layers with a
nearest-neighbor upsample (``view`` + ``expand`` + ``reshape``, MPS-native) plus
``Conv3d``, matching the approach in ``nnUNetTrainer_MPS3D_500epochs``.

This is a smoke/compatibility path, not a claim of numerical equivalence with
the original ``ConvTranspose3d`` architecture.
"""

from __future__ import annotations

import os
from typing import Any

# Must be set before ``import torch`` so PyTorch reads it at backend init.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

MODEL_PATCH_NAME = "ConvTranspose3d_to_InterpConv3d"
MODEL_PATCH_SCOPE = "decoder_upsampling"


def enable_mps_cpu_fallback() -> bool:
    """Ensure PyTorch MPS→CPU fallback is enabled. Returns whether it is active."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


def mps_is_available(torch_module: Any | None = None) -> bool:
    torch_module = torch_module or torch
    mps = getattr(torch_module.backends, "mps", None)
    return mps is not None and mps.is_available()


def resolve_training_device(torch_module: Any, requested: str) -> tuple[Any, dict[str, Any]]:
    """Resolve ``device: auto|cpu|cuda|mps`` and return audit metadata."""
    requested_norm = str(requested or "auto").lower()
    fallback_enabled = enable_mps_cpu_fallback()

    if requested_norm != "auto":
        device = torch_module.device(requested_norm)
    elif torch_module.cuda.is_available():
        device = torch_module.device("cuda")
    elif mps_is_available(torch_module):
        device = torch_module.device("mps")
    else:
        device = torch_module.device("cpu")

    runtime: dict[str, Any] = {
        "device_requested": requested_norm,
        "device_selected": device.type,
        "mps_available": mps_is_available(torch_module),
        "mps_fallback_enabled": fallback_enabled,
        "mps_convtranspose_patched": False,
        "mps_convtranspose_replaced_count": 0,
        "model_patch": None,
        "patch_scope": None,
        "native_mps_claim": False,
    }
    return device, runtime


def record_mps_convtranspose_patch(runtime: dict[str, Any], replaced_count: int) -> dict[str, Any]:
    """Update runtime metadata after applying the ConvTranspose3d replacement patch."""
    runtime = dict(runtime)
    runtime["mps_convtranspose_replaced_count"] = int(replaced_count)
    runtime["mps_convtranspose_patched"] = replaced_count > 0
    if replaced_count > 0:
        runtime["model_patch"] = MODEL_PATCH_NAME
        runtime["patch_scope"] = MODEL_PATCH_SCOPE
    runtime["native_mps_claim"] = False
    return runtime


class InterpConv3d(nn.Module):
    """Drop-in replacement for ``ConvTranspose3d`` in 3D UNet decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Any = 1,
        padding_mode: str = "zeros",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(stride, int):
            scale_factor = (int(stride), int(stride), int(stride))
        else:
            scale_factor = tuple(int(s) for s in stride)
        if not all(s >= 1 and int(s) == s for s in scale_factor):
            raise ValueError(f"InterpConv3d expects integer scale factors per axis, got {scale_factor}")
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bool(bias),
            groups=int(groups),
        )

    @staticmethod
    def _upsample_nearest_3d(x: torch.Tensor, scale_factor: tuple[int, int, int]) -> torch.Tensor:
        sd, sh, sw = scale_factor
        if (sd, sh, sw) == (1, 1, 1):
            return x
        n, c, d, h, w = x.shape
        return x.view(n, c, d, 1, h, 1, w, 1).expand(n, c, d, sd, h, sh, w, sw).reshape(n, c, d * sd, h * sh, w * sw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._upsample_nearest_3d(x, self.scale_factor)
        return self.conv(x)


def replace_convtranspose3d_modules(module: nn.Module) -> int:
    """Recursively replace ``ConvTranspose3d`` with ``InterpConv3d``."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose3d):
            replacement = InterpConv3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                output_padding=child.output_padding,
                groups=child.groups,
                bias=child.bias is not None,
                dilation=child.dilation,
                padding_mode=child.padding_mode,
            )
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += replace_convtranspose3d_modules(child)
    return replaced


def apply_mps_safe_convtranspose_patch(model: nn.Module) -> int:
    """Patch a model so its 3D transposed convolutions run on Apple MPS."""
    return replace_convtranspose3d_modules(model)
