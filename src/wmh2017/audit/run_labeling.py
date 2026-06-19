"""Pure helpers for run mode labels and audit claim wording (torch-free)."""

from __future__ import annotations


def run_purpose_for_mode(mode: str) -> str:
    if mode == "full":
        return "wmh2017_monai_full_training"
    return "wmh2017_monai_smoke_training"


def completion_message(mode: str, run_id: str) -> str:
    label = "full" if mode == "full" else "smoke"
    return f"Completed MONAI {label} training run_id={run_id}"


def checkpoint_filename(mode: str) -> str:
    return "model_best.pt" if mode == "full" else "model_smoke.pt"


def mps_execution_claim(device_type: str, *, patched: bool, mode: str) -> str:
    if device_type != "mps":
        return "standard device path without MPS ConvTranspose3d patch"
    if not patched:
        return "MPS selected without ConvTranspose3d patch"
    scope = "full training" if mode == "full" else "smoke"
    return f"MPS-compatible patched {scope}; not native-MPS equivalence with ConvTranspose3d"


def architecture_parity_block(
    *,
    device_type: str,
    patched: bool,
    patch_name: str = "ConvTranspose3d_to_InterpConv3d",
) -> dict[str, bool | str]:
    substituted = device_type == "mps" and patched
    return {
        "mps_convtranspose_substituted": substituted,
        "comparable_to_native_convtranspose": not substituted,
        "patch_name": patch_name if substituted else "",
    }
