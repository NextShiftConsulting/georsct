"""Unit tests for raster differencing analysis."""

import numpy as np

from georsct.analysis.raster_diff import (
    change_magnitude_map,
    compute_frame_diff,
    find_hotspots,
    temporal_profile,
)


class TestComputeFrameDiff:

    def test_identical_frames_zero_diff(self):
        frame = np.random.rand(32, 32)
        result = compute_frame_diff(frame, frame)
        assert result.mean_abs_diff == 0.0
        assert result.max_abs_diff == 0.0
        assert result.change_fraction == 0.0

    def test_different_frames_positive_diff(self):
        a = np.zeros((32, 32))
        b = np.ones((32, 32))
        result = compute_frame_diff(a, b, threshold=0.5)
        assert result.mean_abs_diff == 1.0
        assert result.change_fraction == 1.0

    def test_diff_map_shape(self):
        a = np.random.rand(3, 32, 32)
        b = np.random.rand(3, 32, 32)
        result = compute_frame_diff(a, b)
        assert result.diff_map.shape == (32, 32)


class TestTemporalProfile:

    def test_3d_sequence(self):
        seq = np.arange(40).reshape(10, 2, 2).astype(float)
        profile = temporal_profile(seq, row=0, col=1)
        assert profile.shape == (10,)

    def test_4d_sequence(self):
        seq = np.random.rand(5, 3, 8, 8)
        profile = temporal_profile(seq, row=4, col=4, channel=1)
        assert profile.shape == (5,)
        np.testing.assert_array_equal(profile, seq[:, 1, 4, 4])


class TestChangeMagnitudeMap:

    def test_static_sequence_zero_change(self):
        frame = np.random.rand(16, 16)
        seq = np.stack([frame] * 5)
        result = change_magnitude_map(seq)
        assert result.shape == (16, 16)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_changing_sequence_positive(self):
        seq = np.random.rand(5, 16, 16)
        result = change_magnitude_map(seq)
        assert result.shape == (16, 16)
        assert result.sum() > 0


class TestFindHotspots:

    def test_returns_correct_count(self):
        change_map = np.random.rand(16, 16)
        spots = find_hotspots(change_map, n_hotspots=3)
        assert len(spots) == 3

    def test_sorted_descending(self):
        change_map = np.random.rand(16, 16)
        spots = find_hotspots(change_map, n_hotspots=5)
        mags = [s[2] for s in spots]
        assert mags == sorted(mags, reverse=True)

    def test_finds_known_max(self):
        change_map = np.zeros((8, 8))
        change_map[3, 5] = 10.0
        spots = find_hotspots(change_map, n_hotspots=1)
        assert spots[0][0] == 3
        assert spots[0][1] == 5
