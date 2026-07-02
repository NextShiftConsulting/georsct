"""Unit tests for raster loading utilities."""

import pytest
import torch
import numpy as np

from georsct.loaders.raster import (
    center_crop,
    make_synthetic_raster,
    make_synthetic_raster_pair,
    make_synthetic_raster_sequence,
    numpy_to_tensor,
)


class TestNumpyToTensor:

    def test_2d_adds_channel(self):
        arr = np.random.rand(32, 32)
        t = numpy_to_tensor(arr, normalize=False)
        assert t.shape == (1, 32, 32)

    def test_3d_keeps_shape(self):
        arr = np.random.rand(3, 32, 32)
        t = numpy_to_tensor(arr, normalize=False)
        assert t.shape == (3, 32, 32)

    def test_normalization(self):
        arr = np.array([[0.0, 100.0], [50.0, 200.0]])
        t = numpy_to_tensor(arr, normalize=True)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_dtype_float32(self):
        arr = np.random.rand(32, 32).astype(np.float64)
        t = numpy_to_tensor(arr)
        assert t.dtype == torch.float32


class TestCenterCrop:

    def test_basic_crop(self):
        t = torch.randn(3, 64, 64)
        cropped = center_crop(t, 32)
        assert cropped.shape == (3, 32, 32)

    def test_crop_center_values(self):
        t = torch.zeros(1, 10, 10)
        t[0, 5, 5] = 1.0
        cropped = center_crop(t, 4)
        assert cropped.shape == (1, 4, 4)
        assert cropped[0, 2, 2] == 1.0

    def test_too_small_raises(self):
        t = torch.randn(3, 16, 16)
        with pytest.raises(ValueError, match="smaller than crop size"):
            center_crop(t, 32)


class TestSyntheticData:

    def test_raster_shape(self):
        x = make_synthetic_raster(4, channels=2, height=32, width=32, seed=0)
        assert x.shape == (4, 2, 32, 32)

    def test_raster_range(self):
        x = make_synthetic_raster(4, seed=0)
        assert x.min() >= 0.0
        assert x.max() <= 1.0

    def test_raster_reproducible(self):
        x1 = make_synthetic_raster(2, seed=7)
        x2 = make_synthetic_raster(2, seed=7)
        torch.testing.assert_close(x1, x2)

    def test_pair_shape(self):
        x, t = make_synthetic_raster_pair(3, channels=1, height=32, width=32, seed=0)
        assert x.shape == (3, 2, 1, 32, 32)
        assert t.shape == (3, 2)

    def test_pair_has_change(self):
        x, _ = make_synthetic_raster_pair(1, change_intensity=1.0, seed=0)
        diff = (x[0, 1] - x[0, 0]).abs()
        assert diff.max() > 0.01

    def test_sequence_shape(self):
        x, t = make_synthetic_raster_sequence(2, n_frames=8, seed=0)
        assert x.shape == (2, 8, 1, 64, 64)
        assert t.shape == (2, 8)

    def test_sequence_timestamps_sorted(self):
        _, t = make_synthetic_raster_sequence(4, n_frames=10, seed=0)
        for b in range(4):
            diffs = t[b, 1:] - t[b, :-1]
            assert (diffs >= 0).all()
