import numpy as np

from wmh2017.data.preprocessing import normalize_nonzero_channelwise


def test_normalize_nonzero_single_volume_excludes_zero_background():
    x = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 2, 2)
    out = normalize_nonzero_channelwise(x)

    nonzero = out[x != 0]
    assert np.isclose(float(nonzero.mean()), 0.0, atol=1e-6)
    assert np.isclose(float(nonzero.std()), 1.0, atol=1e-6)
    assert out.dtype == np.float32


def test_normalize_nonzero_channelwise_keeps_channels_independent():
    x = np.zeros((2, 1, 2, 2), dtype=np.float32)
    x[0, 0] = np.array([[0, 1], [2, 3]], dtype=np.float32)
    x[1, 0] = np.array([[0, 10], [20, 30]], dtype=np.float32)

    out = normalize_nonzero_channelwise(x)

    for channel_idx in range(2):
        nonzero = out[channel_idx][x[channel_idx] != 0]
        assert np.isclose(float(nonzero.mean()), 0.0, atol=1e-6)
        assert np.isclose(float(nonzero.std()), 1.0, atol=1e-6)
