"""Unit tests for spatial raster encoders."""

import pytest
import torch

from georsct.encoders.spatial_base import SpatialEncoderOutput
from georsct.encoders.patch_conv import PatchConvEncoder
from georsct.encoders.siamese_diff import SiameseDiffEncoder
from georsct.encoders.raster_seq import RasterSeqEncoder


BATCH = 2
IN_CHANNELS = 3
HIDDEN_DIM = 32
H, W = 64, 64


@pytest.fixture
def single_raster():
    torch.manual_seed(42)
    return torch.randn(BATCH, IN_CHANNELS, H, W)


@pytest.fixture
def raster_pair():
    torch.manual_seed(42)
    return torch.randn(BATCH, 2, IN_CHANNELS, H, W)


@pytest.fixture
def raster_sequence():
    torch.manual_seed(42)
    return torch.randn(BATCH, 4, IN_CHANNELS, H, W)


class TestPatchConvEncoder:

    def test_output_type(self, single_raster):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(single_raster)
        assert isinstance(out, SpatialEncoderOutput)

    def test_hidden_shape(self, single_raster):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(single_raster)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)

    def test_feature_map_shape(self, single_raster):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(single_raster)
        assert out.feature_map.shape[0] == BATCH
        assert out.feature_map.shape[1] == 512  # ResNet18 layer4

    def test_finite_output(self, single_raster):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(single_raster)
        assert torch.isfinite(out.hidden).all()

    def test_single_channel(self):
        enc = PatchConvEncoder(1, HIDDEN_DIM)
        x = torch.randn(BATCH, 1, H, W)
        out = enc(x)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)

    def test_gradients(self, single_raster):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(single_raster)
        out.hidden.sum().backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestSiameseDiffEncoder:

    def test_output_type(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(raster_pair)
        assert isinstance(out, SpatialEncoderOutput)

    def test_hidden_shape(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(raster_pair)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)

    def test_diff_feature_map(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(raster_pair)
        assert out.feature_map.shape[0] == BATCH
        assert out.feature_map.shape[1] == 512

    def test_finite_output(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(raster_pair)
        assert torch.isfinite(out.hidden).all()

    def test_with_timestamps(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        timestamps = torch.tensor([[0.0, 24.0], [0.0, 48.0]])
        out = enc(raster_pair, timestamps=timestamps)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)
        assert torch.isfinite(out.hidden).all()

    def test_identical_frames_small_diff(self):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        enc.eval()
        frame = torch.randn(1, IN_CHANNELS, H, W)
        x = torch.stack([frame, frame], dim=1)
        with torch.no_grad():
            out = enc(x)
        assert out.feature_map.abs().max() < 1e-5

    def test_gradients(self, raster_pair):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        out = enc(raster_pair)
        out.hidden.sum().backward()
        has_grads = any(p.grad is not None for p in enc.parameters() if p.requires_grad)
        assert has_grads


class TestRasterSeqEncoder:

    def test_output_type(self, raster_sequence):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        out = enc(raster_sequence)
        assert isinstance(out, SpatialEncoderOutput)

    def test_hidden_shape(self, raster_sequence):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        out = enc(raster_sequence)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)

    def test_finite_output(self, raster_sequence):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        out = enc(raster_sequence)
        assert torch.isfinite(out.hidden).all()

    def test_with_timestamps(self, raster_sequence):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        timestamps = torch.sort(torch.rand(BATCH, 4), dim=1)[0] * 48.0
        out = enc(raster_sequence, timestamps=timestamps)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)
        assert torch.isfinite(out.hidden).all()

    def test_single_frame_sequence(self):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        x = torch.randn(BATCH, 1, IN_CHANNELS, H, W)
        out = enc(x)
        assert out.hidden.shape == (BATCH, HIDDEN_DIM)

    def test_gradients(self, raster_sequence):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        out = enc(raster_sequence)
        out.hidden.sum().backward()
        has_grads = any(p.grad is not None for p in enc.parameters() if p.requires_grad)
        assert has_grads
