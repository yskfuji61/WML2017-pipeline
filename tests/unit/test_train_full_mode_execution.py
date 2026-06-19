"""Full-mode training execution smoke (synthetic tensors; no dataset I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.requires_torch
def test_full_mode_synthetic_training_step_and_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("monai")
    import torch
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    device = torch.device("cpu")
    model = model.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    global_step = 0
    for _step in range(2):
        images = torch.randn(1, 1, 96, 96, 96, device=device)
        labels = torch.randint(0, 2, (1, 1, 96, 96, 96), device=device).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        global_step += 1

    checkpoint_path = tmp_path / "model_best.pt"
    torch.save(  # nosec B614 — local synthetic checkpoint for unit test only
        {
            "model_state_dict": model.state_dict(),
            "global_step": global_step,
            "best_val_dice": 0.0,
            "claim_boundary": "synthetic full-mode execution test only",
        },
        checkpoint_path,
    )
    assert checkpoint_path.exists()

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec B614
    model.load_state_dict(state["model_state_dict"])
    assert int(state["global_step"]) == 2
