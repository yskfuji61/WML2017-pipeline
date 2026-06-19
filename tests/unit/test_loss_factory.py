from __future__ import annotations

import torch

from wmh2017.training.loss_factory import build_loss


class _FakeMonaiLoss:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self


def test_build_loss_defaults_to_monai_dice_ce():
    monai = {"DiceCELoss": _FakeMonaiLoss}
    loss = build_loss({}, monai)
    out = loss(torch.zeros(1, 2, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4, dtype=torch.long))
    assert isinstance(out, _FakeMonaiLoss)


def test_build_loss_tversky_focal_finite():
    monai = {"DiceCELoss": _FakeMonaiLoss}
    loss = build_loss({"loss": {"name": "tversky_focal", "alpha": 0.3, "beta": 0.7}}, monai)
    logits = torch.zeros(1, 2, 1, 4, 4, dtype=torch.float32)
    logits[:, 1, :, 2, 2] = 2.0
    targets = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    targets[:, :, 2, 2] = 1
    value = loss(logits, targets)
    assert torch.isfinite(value)


def test_tversky_penalizes_false_negatives_more():
    monai = {"DiceCELoss": _FakeMonaiLoss}
    fn_heavy = build_loss({"loss": {"name": "tversky", "alpha": 0.3, "beta": 0.7}}, monai)
    logits = torch.zeros(1, 2, 1, 4, 4)
    logits[:, 1, :, 1:3, 1:3] = 5.0
    targets = torch.zeros(1, 1, 4, 4, dtype=torch.long)
    targets[:, :, 1:3, 1:3] = 1
    under = logits.clone()
    under[:, 1, :, 1:3, 1:3] = -5.0
    assert fn_heavy(under, targets) > fn_heavy(logits, targets)
