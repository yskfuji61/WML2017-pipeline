"""ConvNeXt 2.5D best checkpoint must be val_loss_proxy, not Dice/recall best."""

from __future__ import annotations

import ast
from pathlib import Path

from wmh2017.training.train_convnext_25d import CONVNEXT_SELECTION_POLICY

_SRC = Path(__file__).resolve().parents[2] / "src/wmh2017/training/train_convnext_25d.py"


def test_selection_policy_is_val_loss_proxy_min() -> None:
    assert CONVNEXT_SELECTION_POLICY["selection_metric"] == "val_loss_proxy"
    assert CONVNEXT_SELECTION_POLICY["selection_mode"] == "min"


def test_semantics_string_disclaims_dice_and_recall() -> None:
    semantics = CONVNEXT_SELECTION_POLICY["checkpoint_semantics"].lower()
    assert "not best dice" in semantics
    assert "not best lesion recall" in semantics


def test_primary_checkpoint_name_reveals_loss_proxy() -> None:
    source = _SRC.read_text(encoding="utf-8")
    # Primary best checkpoint must encode the selection semantics in its filename.
    assert "model_best_val_loss_proxy.pt" in source
    # model_best.pt may only exist as a legacy alias.
    tree = ast.parse(source)
    legacy_alias_marked = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "legacy" in node.value.lower()
        and "model_best.pt" in node.value
        for node in ast.walk(tree)
    )
    assert legacy_alias_marked
