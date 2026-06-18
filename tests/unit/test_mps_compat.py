import os

import pytest
import torch
import torch.nn as nn

from wmh2017.training.mps_compat import (
    apply_mps_safe_convtranspose_patch,
    enable_mps_cpu_fallback,
    record_mps_convtranspose_patch,
    replace_convtranspose3d_modules,
    resolve_training_device,
)


def test_enable_mps_cpu_fallback_sets_env():
    enable_mps_cpu_fallback()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_replace_convtranspose3d_modules_replaces_all_layers():
    model = nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.ConvTranspose3d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
    )
    replaced = replace_convtranspose3d_modules(model)
    assert replaced == 1
    assert not any(isinstance(m, nn.ConvTranspose3d) for m in model.modules())


def test_record_mps_convtranspose_patch_sets_audit_fields():
    runtime = record_mps_convtranspose_patch(
        {
            "device_requested": "auto",
            "device_selected": "mps",
            "mps_available": True,
            "mps_fallback_enabled": True,
            "mps_convtranspose_patched": False,
            "mps_convtranspose_replaced_count": 0,
            "model_patch": None,
            "patch_scope": None,
            "native_mps_claim": False,
        },
        replaced_count=2,
    )
    assert runtime["mps_convtranspose_patched"] is True
    assert runtime["mps_convtranspose_replaced_count"] == 2
    assert runtime["model_patch"] == "ConvTranspose3d_to_InterpConv3d"
    assert runtime["patch_scope"] == "decoder_upsampling"
    assert runtime["native_mps_claim"] is False


def test_resolve_training_device_auto_prefers_cuda_then_mps_then_cpu():
    device, runtime = resolve_training_device(torch, "auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"
    assert runtime["device_requested"] == "auto"
    assert runtime["device_selected"] == device.type
    assert runtime["mps_fallback_enabled"] is True
    assert runtime["native_mps_claim"] is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_monai_unet_runs_on_mps_after_patch():
    pytest.importorskip("monai")
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
    )
    patched = apply_mps_safe_convtranspose_patch(model)
    assert patched >= 1
    model = model.to("mps")
    x = torch.randn(1, 1, 32, 32, 32, device="mps")
    y = model(x)
    assert y.shape == (1, 2, 32, 32, 32)
    assert y.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_monai_unet_mps_training_step_after_patch():
    pytest.importorskip("monai")
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
    )
    assert apply_mps_safe_convtranspose_patch(model) >= 1
    model = model.to("mps")
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    images = torch.randn(1, 1, 32, 32, 32, device="mps")
    labels = torch.randint(0, 2, (1, 1, 32, 32, 32), device="mps")
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
