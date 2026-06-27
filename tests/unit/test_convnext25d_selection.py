"""T1-R1: best-validation-Dice checkpoint selection for the 2.5D ConvNeXt trainer.

Pure unit tests (no training run) covering the default-off ``training.checkpoint_selection``
support: mode resolution/validation, the selection comparator, the micro-Dice helper, and two
static-source guards (no heldout/test path; export filename contract unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh2017.training import train_convnext_25d as t

SRC = Path(t.__file__)
EXPORT_SRC = Path(t.__file__).parent.parent / "inference" / "export_convnext_probabilities.py"


# --- Test 1: key absent keeps val_loss_proxy (default-off) ---
def test_key_absent_keeps_val_loss_proxy():
    assert t.resolve_checkpoint_selection({}) == "val_loss_proxy"
    assert t.resolve_checkpoint_selection({"checkpoint_selection": "val_loss_proxy"}) == "val_loss_proxy"
    assert t.resolve_checkpoint_selection({"checkpoint_selection": "best_val_dice"}) == "best_val_dice"
    # Case/whitespace tolerant but still a known mode.
    assert t.resolve_checkpoint_selection({"checkpoint_selection": " Best_Val_Dice "}) == "best_val_dice"


# --- Test 2: best_val_dice selects a higher Dice epoch even when val_loss_proxy is worse ---
def test_best_val_dice_selects_higher_dice_even_if_loss_worse():
    # Per-epoch (val_loss, val_dice): epoch0 has the best Dice but the worse loss.
    epochs = [
        {"epoch": 0, "val_loss": 0.50, "val_dice": 0.60},
        {"epoch": 1, "val_loss": 0.40, "val_dice": 0.55},
    ]

    def stream_select(mode: str) -> int:
        best = float("-inf") if mode == "best_val_dice" else float("inf")
        chosen = -1
        for row in epochs:
            cand = row["val_dice"] if mode == "best_val_dice" else row["val_loss"]
            if t.is_better_selection(mode, cand, best):
                best = cand
                chosen = row["epoch"]
        return chosen

    assert stream_select("val_loss_proxy") == 1  # min loss
    assert stream_select("best_val_dice") == 0  # max dice
    assert stream_select("val_loss_proxy") != stream_select("best_val_dice")


# --- Test 3: deterministic tie-breaking (strictly-better => earliest epoch kept) ---
def test_tie_break_is_deterministic_earliest_epoch():
    # Equal candidate never replaces the incumbent => the earliest epoch is retained.
    assert t.is_better_selection("best_val_dice", 0.7, 0.7) is False
    assert t.is_better_selection("val_loss_proxy", 0.3, 0.3) is False
    # Strictly-better still wins.
    assert t.is_better_selection("best_val_dice", 0.71, 0.70) is True
    assert t.is_better_selection("val_loss_proxy", 0.29, 0.30) is True

    epochs = [
        {"epoch": 0, "val_dice": 0.65},
        {"epoch": 1, "val_dice": 0.65},  # tie -> must NOT replace epoch 0
        {"epoch": 2, "val_dice": 0.65},
    ]
    best = float("-inf")
    chosen = -1
    for row in epochs:
        if t.is_better_selection("best_val_dice", row["val_dice"], best):
            best = row["val_dice"]
            chosen = row["epoch"]
    assert chosen == 0


# --- Test 4: invalid mode raises a clear error ---
def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="checkpoint_selection"):
        t.resolve_checkpoint_selection({"checkpoint_selection": "bogus"})


# --- Test 5: no heldout/test path + micro_dice numeric contract ---
def test_no_heldout_path_and_micro_dice_numeric():
    source = SRC.read_text(encoding="utf-8")
    # Trainer constructs WmhVolumeDataset only with the train/val assigned splits.
    assert "WmhVolumeDataset(" in source
    assert '"train"' in source and '"val"' in source
    assert '"heldout_eval"' not in source
    assert '"heldout"' not in source
    assert '"test"' not in source

    # micro_dice == 2*tp / (2*tp + fp + fn): tp=10, fp=2, fn=4 -> 20/26.
    tp, fp, fn = 10.0, 2.0, 4.0
    inter, pred_sum, gt_sum = tp, tp + fp, tp + fn
    assert t.micro_dice(inter, pred_sum, gt_sum) == pytest.approx(20.0 / 26.0, rel=1e-6)
    # Empty prediction and empty ground truth -> perfect (empty-score convention).
    assert t.micro_dice(0.0, 0.0, 0.0) == 1.0
    # Perfect overlap -> 1.0.
    assert t.micro_dice(5.0, 5.0, 5.0) == pytest.approx(1.0, rel=1e-6)


# --- Test 6: export filename contract unchanged ({case_id}.npz) ---
def test_export_filename_contract_unchanged():
    export_source = EXPORT_SRC.read_text(encoding="utf-8")
    assert "f\"{sample['case_id']}.npz\"" in export_source
