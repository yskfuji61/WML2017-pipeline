from __future__ import annotations

from wmh2017.audit.run_labeling import (
    architecture_parity_block,
    checkpoint_filename,
    completion_message,
    mps_execution_claim,
    run_purpose_for_mode,
)


def test_run_purpose_for_mode() -> None:
    assert run_purpose_for_mode("full") == "wmh2017_monai_full_training"
    assert run_purpose_for_mode("smoke") == "wmh2017_monai_smoke_training"
    assert run_purpose_for_mode("SMOKE") == "wmh2017_monai_smoke_training"


def test_completion_message() -> None:
    assert completion_message("full", "run_x") == "Completed MONAI full training run_id=run_x"
    assert completion_message("smoke", "run_y") == "Completed MONAI smoke training run_id=run_y"


def test_checkpoint_filename() -> None:
    assert checkpoint_filename("full") == "model_best.pt"
    assert checkpoint_filename("smoke") == "model_smoke.pt"


def test_mps_execution_claim_full_mode() -> None:
    claim = mps_execution_claim("mps", patched=True, mode="full")
    assert "full training" in claim
    assert "smoke" not in claim.lower() or "full training" in claim


def test_mps_execution_claim_smoke_mode() -> None:
    claim = mps_execution_claim("mps", patched=True, mode="smoke")
    assert "smoke" in claim


def test_mps_execution_claim_cpu() -> None:
    claim = mps_execution_claim("cpu", patched=False, mode="full")
    assert "standard device path" in claim


def test_architecture_parity_mps_patched() -> None:
    block = architecture_parity_block(device_type="mps", patched=True)
    assert block["mps_convtranspose_substituted"] is True
    assert block["comparable_to_native_convtranspose"] is False
    assert block["patch_name"] == "ConvTranspose3d_to_InterpConv3d"


def test_architecture_parity_cpu() -> None:
    block = architecture_parity_block(device_type="cpu", patched=False)
    assert block["mps_convtranspose_substituted"] is False
    assert block["comparable_to_native_convtranspose"] is True
    assert block["patch_name"] == ""
