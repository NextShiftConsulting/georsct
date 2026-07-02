"""Unit tests for the spatial-to-embedding adapter."""

import pytest
import torch
import numpy as np

from georsct.encoders.patch_conv import PatchConvEncoder
from georsct.encoders.siamese_diff import SiameseDiffEncoder
from georsct.adapters.spatial_to_embedding import SpatialToEmbedding, SpatialPipeline


BATCH = 2
IN_CHANNELS = 3
HIDDEN_DIM = 32
EMBEDDING_DIM = 64
H, W = 64, 64


@pytest.fixture(params=["patch_conv", "siamese_diff"])
def encoder(request):
    if request.param == "patch_conv":
        return PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
    return SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)


@pytest.fixture
def sample_input(encoder):
    torch.manual_seed(42)
    if isinstance(encoder, SiameseDiffEncoder):
        return torch.randn(BATCH, 2, IN_CHANNELS, H, W)
    return torch.randn(BATCH, IN_CHANNELS, H, W)


class TestAdapterShapes:

    def test_forward_shape(self, encoder, sample_input):
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        enc_out = encoder(sample_input)
        emb = adapter(enc_out)
        assert emb.shape == (BATCH, EMBEDDING_DIM)

    def test_p19_default_dim(self):
        adapter = SpatialToEmbedding(HIDDEN_DIM)
        assert adapter.embedding_dim == 64

    def test_to_numpy_dtype(self, encoder, sample_input):
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        enc_out = encoder(sample_input)
        arr = adapter.to_numpy(enc_out)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (BATCH, EMBEDDING_DIM)

    def test_to_numpy_finite(self, encoder, sample_input):
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        enc_out = encoder(sample_input)
        arr = adapter.to_numpy(enc_out)
        assert np.isfinite(arr).all()


class TestPipeline:

    def test_encode_returns_numpy(self, encoder, sample_input):
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipeline = SpatialPipeline(encoder, adapter)
        result = pipeline.encode(sample_input)
        assert isinstance(result, np.ndarray)
        assert result.shape == (BATCH, EMBEDDING_DIM)

    def test_encode_deterministic(self, encoder, sample_input):
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipeline = SpatialPipeline(encoder, adapter)
        r1 = pipeline.encode(sample_input)
        r2 = pipeline.encode(sample_input)
        np.testing.assert_array_equal(r1, r2)
