"""Config-gated LR scheduler: default off (backward compatible), opt-in by config."""

from __future__ import annotations

import pytest


def _tiny_optimizer(torch):
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=0.1)


def test_no_scheduler_by_default() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    scheduler, info = build_lr_scheduler(torch, opt, {}, max_epochs=10)
    assert scheduler is None
    assert info["enabled"] is False


def test_explicit_none_disables() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    scheduler, info = build_lr_scheduler(torch, opt, {"lr_scheduler": {"name": "none"}}, max_epochs=10)
    assert scheduler is None
    assert info["enabled"] is False


def test_cosine_scheduler_decays_lr() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    scheduler, info = build_lr_scheduler(
        torch, opt, {"lr_scheduler": {"name": "cosine", "eta_min": 0.0}}, max_epochs=10
    )
    assert info == {"enabled": True, "name": "cosine", "t_max": 10, "eta_min": 0.0}
    start_lr = opt.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert opt.param_groups[0]["lr"] < start_lr


def test_poly_scheduler_builds() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    scheduler, info = build_lr_scheduler(torch, opt, {"lr_scheduler": {"name": "poly", "power": 0.9}}, max_epochs=10)
    assert info["name"] == "poly"
    start_lr = opt.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert opt.param_groups[0]["lr"] < start_lr


def test_unknown_scheduler_raises() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    with pytest.raises(ValueError):
        build_lr_scheduler(torch, opt, {"lr_scheduler": {"name": "bogus"}}, max_epochs=10)


def test_warmup_default_off_unchanged() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    scheduler, info = build_lr_scheduler(
        torch, opt, {"lr_scheduler": {"name": "cosine", "eta_min": 0.0}}, max_epochs=20
    )
    assert info == {"enabled": True, "name": "cosine", "t_max": 20, "eta_min": 0.0}
    assert "warmup_epochs" not in info


def test_cosine_warmup_ramps_then_decays() -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)  # peak lr 0.1
    peak = 0.1
    scheduler, info = build_lr_scheduler(
        torch, opt, {"warmup_epochs": 5, "lr_scheduler": {"name": "cosine", "eta_min": 0.0}}, max_epochs=20
    )
    assert info["name"] == "cosine_warmup"
    assert info["warmup_epochs"] == 5
    lrs = [opt.param_groups[0]["lr"]]  # epoch 0 (start of warmup)
    for _ in range(19):
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Warmup: starts well below peak and ramps up across the first 5 epochs to ~peak.
    assert lrs[0] < peak
    assert lrs[4] > lrs[0]
    assert lrs[5] >= 0.9 * peak
    # After warmup: cosine decay (later epochs strictly below the peak).
    assert lrs[10] < lrs[5]
    assert lrs[-1] < lrs[10]


@pytest.mark.parametrize("bad", [-1, 20, 25])
def test_invalid_warmup_epochs_raises(bad) -> None:
    torch = pytest.importorskip("torch")
    from wmh2017.training.train_monai import build_lr_scheduler

    opt = _tiny_optimizer(torch)
    with pytest.raises(ValueError, match="warmup_epochs"):
        build_lr_scheduler(torch, opt, {"warmup_epochs": bad, "lr_scheduler": {"name": "cosine"}}, max_epochs=20)
