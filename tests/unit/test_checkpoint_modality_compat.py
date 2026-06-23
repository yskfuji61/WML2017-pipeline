from __future__ import annotations

import pytest

from wmh2017.config.training_config import InputModality
from wmh2017.models.factory import assert_checkpoint_modality_compat

FLAIR = (InputModality(name="image", manifest_key="flair_pre_path"),)
FLAIR_T1 = (
    InputModality(name="flair", manifest_key="flair_pre_path"),
    InputModality(name="t1", manifest_key="t1_pre_path"),
)


def test_matching_counts_pass():
    assert_checkpoint_modality_compat(
        checkpoint_modalities=[{"name": "flair"}, {"name": "t1"}],
        config_modalities=FLAIR_T1,
    )


def test_mismatched_counts_raise():
    with pytest.raises(ValueError, match="refusing to load mismatched weights"):
        assert_checkpoint_modality_compat(
            checkpoint_modalities=[{"name": "image"}],
            config_modalities=FLAIR_T1,
        )


def test_legacy_checkpoint_without_metadata_is_single_channel():
    # None metadata is treated as one channel -> compatible with FLAIR-only config.
    assert_checkpoint_modality_compat(checkpoint_modalities=None, config_modalities=FLAIR)
    # ...but incompatible with a 2-channel config.
    with pytest.raises(ValueError, match="1 input channel"):
        assert_checkpoint_modality_compat(checkpoint_modalities=None, config_modalities=FLAIR_T1)
