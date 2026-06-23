from __future__ import annotations

from typing import Any

from wmh2017.training.transforms import StackModalitiesd, build_monai_transforms


class _Op:
    """Records the transform name and kwargs it was constructed with."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs


def _fake_monai() -> dict[str, Any]:
    names = [
        "LoadImaged",
        "EnsureChannelFirstd",
        "Lambdad",
        "ResizeWithPadOrCropd",
        "EnsureTyped",
    ]
    monai: dict[str, Any] = {name: (lambda name: lambda **kw: _Op(name, **kw))(name) for name in names}
    monai["Compose"] = lambda ops: ops  # return the op list directly for inspection
    return monai


def test_flair_only_pipeline_unchanged():
    ops = build_monai_transforms(_fake_monai(), [16, 16, 16], train=False)
    load = ops[0]
    assert isinstance(load, _Op) and load.name == "LoadImaged"
    assert load.kwargs["keys"] == ["image", "label"]
    # No modality stacking for the single-channel default.
    assert not any(isinstance(op, StackModalitiesd) for op in ops)


def test_multimodal_pipeline_inserts_single_stack_op():
    ops = build_monai_transforms(_fake_monai(), [16, 16, 16], train=False, input_keys=("flair", "t1"))
    load = ops[0]
    assert load.kwargs["keys"] == ["flair", "t1", "label"]
    stack_ops = [op for op in ops if isinstance(op, StackModalitiesd)]
    assert len(stack_ops) == 1
    assert stack_ops[0].keys == ("flair", "t1")
    assert stack_ops[0].output_key == "image"


def test_stack_modalities_concatenates_channels():
    import numpy as np

    data = {
        "flair": np.ones((1, 2, 2, 2), dtype=np.float32),
        "t1": np.full((1, 2, 2, 2), 2.0, dtype=np.float32),
        "label": np.zeros((1, 2, 2, 2), dtype=np.int64),
    }
    out = StackModalitiesd(keys=("flair", "t1"))(data)
    assert out["image"].shape == (2, 2, 2, 2)
    assert "t1" not in out
    assert "label" in out
