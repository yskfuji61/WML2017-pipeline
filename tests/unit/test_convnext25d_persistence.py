"""T1-R3: checkpoint persistence + validation probability diagnostics for the 2.5D ConvNeXt trainer.

Pure unit tests (no training run) covering the additive checkpoint-filename/inventory helpers and the
streaming validation-only probability diagnostics. The selected-checkpoint contract is verified to be
unchanged (filenames + alias per mode).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wmh2017.training import train_convnext_25d as t

SRC = Path(t.__file__)


# --- Test 1: model_last.pt always present (both modes) ---
def test_model_last_present_in_both_modes():
    assert t.checkpoint_filenames("val_loss_proxy")["last"] == "model_last.pt"
    assert t.checkpoint_filenames("best_val_dice")["last"] == "model_last.pt"


# --- Test 2: val_loss_proxy selected-checkpoint semantics unchanged ---
def test_val_loss_proxy_selected_unchanged():
    names = t.checkpoint_filenames("val_loss_proxy")
    assert names["selected"] == "model_best_val_loss_proxy.pt"
    assert names["legacy_alias"] == "model_best.pt"
    assert names["loss_proxy_safety"] is None  # no extra safety file in loss mode


# --- Test 3: best_val_dice selected-checkpoint semantics unchanged ---
def test_best_val_dice_selected_unchanged():
    names = t.checkpoint_filenames("best_val_dice")
    assert names["selected"] == "model_best_val_dice.pt"
    assert names["legacy_alias"] == "model_best.pt"


# --- Test 4: dice mode exposes a best-val-loss safety checkpoint target ---
def test_best_val_dice_has_loss_proxy_safety():
    assert t.checkpoint_filenames("best_val_dice")["loss_proxy_safety"] == "model_best_val_loss_proxy.pt"


# --- Test 5: checkpoint inventory keys/roles per mode ---
def test_checkpoint_inventory_keys_and_roles():
    dice_inv = t.build_checkpoint_inventory(
        "best_val_dice", best_epoch=3, best_score=0.42, best_val_loss=0.31, best_loss_epoch=7, last_epoch=29
    )
    assert dice_inv["model_best_val_dice.pt"]["role"] == "selected"
    assert dice_inv["model_best_val_dice.pt"]["epoch"] == 3
    assert dice_inv["model_best_val_dice.pt"]["val_dice"] == 0.42
    assert dice_inv["model_best.pt"]["role"] == "selected_alias"
    assert dice_inv["model_best_val_loss_proxy.pt"]["role"] == "safety_val_loss_proxy"
    assert dice_inv["model_best_val_loss_proxy.pt"]["epoch"] == 7
    assert dice_inv["model_last.pt"]["role"] == "last"
    assert dice_inv["model_last.pt"]["epoch"] == 29

    loss_inv = t.build_checkpoint_inventory(
        "val_loss_proxy", best_epoch=11, best_score=0.30, best_val_loss=0.30, best_loss_epoch=11, last_epoch=29
    )
    assert loss_inv["model_best_val_loss_proxy.pt"]["role"] == "selected"
    assert loss_inv["model_best_val_loss_proxy.pt"]["val_loss_proxy"] == 0.30
    assert loss_inv["model_best.pt"]["role"] == "selected_alias"
    assert loss_inv["model_last.pt"]["role"] == "last"
    # loss mode has no separate safety entry (the selected file IS the loss-proxy checkpoint)
    assert sum(1 for v in loss_inv.values() if v["role"] == "safety_val_loss_proxy") == 0


# --- Test 6: diagnostics numerics on synthetic logits/labels ---
def test_diagnostics_numerics():
    # logits -> probs via sigmoid; build a small deterministic case.
    logits = np.array([[-10.0, 0.0], [2.0, 10.0]], dtype=np.float64)  # sigmoid ~ [[~0,0.5],[0.88,~1]]
    probs = 1.0 / (1.0 + np.exp(-logits))
    label = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    acc = t.new_val_prob_accum()
    t.update_val_prob_accum(acc, probs, label)
    d = t.finalize_val_prob_diagnostics(acc)
    assert d["val_max_prob"] == max(float(probs.max()), 0.0)
    # counts: probs are ~[4.5e-5, 0.5, 0.8808, ~1.0]
    assert d["val_pred_voxels_at_0_15"] == int((probs >= 0.15).sum()) == 3
    assert d["val_pred_voxels_at_0_40"] == int((probs >= 0.40).sum()) == 3
    assert d["val_pred_voxels_at_0_50"] == int((probs >= 0.50).sum()) == 3
    # lesion voxels are at label==1 positions: probs[0,1]=0.5 and probs[1,0]=0.8808...
    lesion_vals = probs[label.astype(bool)]
    assert d["val_lesion_voxel_mean_prob"] == float(lesion_vals.mean())
    assert d["val_lesion_voxel_median_prob"] == float(np.median(lesion_vals))


def test_diagnostics_streaming_matches_single_pass():
    rng = np.random.default_rng(0)
    p1, l1 = rng.random((4, 4)), (rng.random((4, 4)) > 0.5).astype(np.uint8)
    p2, l2 = rng.random((3, 3)), (rng.random((3, 3)) > 0.5).astype(np.uint8)
    acc = t.new_val_prob_accum()
    t.update_val_prob_accum(acc, p1, l1)
    t.update_val_prob_accum(acc, p2, l2)
    d = t.finalize_val_prob_diagnostics(acc)
    assert d["val_max_prob"] == max(float(p1.max()), float(p2.max()))
    assert d["val_pred_voxels_at_0_40"] == int((p1 >= 0.40).sum()) + int((p2 >= 0.40).sum())


# --- Test 7: empty lesion / empty accumulator are safe ---
def test_empty_lesion_and_empty_accum_safe():
    # empty accumulator
    d0 = t.finalize_val_prob_diagnostics(t.new_val_prob_accum())
    assert d0["val_max_prob"] == 0.0
    assert d0["val_pred_voxels_at_0_50"] == 0
    assert d0["val_lesion_voxel_mean_prob"] == 0.0
    assert d0["val_lesion_voxel_median_prob"] == 0.0
    # all-background label -> no lesion voxels
    acc = t.new_val_prob_accum()
    t.update_val_prob_accum(acc, np.full((3, 3), 0.7, dtype=np.float64), np.zeros((3, 3), dtype=np.uint8))
    d = t.finalize_val_prob_diagnostics(acc)
    assert d["val_lesion_voxel_mean_prob"] == 0.0
    assert d["val_lesion_voxel_median_prob"] == 0.0
    assert d["val_pred_voxels_at_0_50"] == 9  # all >= 0.5


# --- Test 8: default path robustness (loss-mode inventory, mismatched sizes) ---
def test_default_path_robustness():
    # loss-mode inventory with default/unused safety epoch must not raise.
    inv = t.build_checkpoint_inventory(
        "val_loss_proxy",
        best_epoch=-1,
        best_score=float("inf"),
        best_val_loss=float("inf"),
        best_loss_epoch=-1,
        last_epoch=0,
    )
    assert "model_last.pt" in inv
    # mismatched prob/label sizes: lesion voxels skipped without error.
    acc = t.new_val_prob_accum()
    t.update_val_prob_accum(acc, np.full((2, 2), 0.3, dtype=np.float64), np.array([1, 0], dtype=np.uint8))
    d = t.finalize_val_prob_diagnostics(acc)
    assert d["val_lesion_voxel_mean_prob"] == 0.0  # size mismatch -> no lesion voxels collected
    assert d["val_pred_voxels_at_0_15"] == 4


# --- Test 9: no heldout/test path referenced by the trainer ---
def test_no_heldout_path_in_source():
    source = SRC.read_text(encoding="utf-8")
    assert "WmhVolumeDataset(" in source
    assert '"train"' in source and '"val"' in source
    assert '"heldout_eval"' not in source
    assert '"heldout"' not in source
    assert '"test"' not in source
