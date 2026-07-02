"""Integration test: full pipeline from raster to rotor-compatible embedding."""

import numpy as np
import torch

from georsct.encoders.patch_conv import PatchConvEncoder
from georsct.encoders.siamese_diff import SiameseDiffEncoder
from georsct.encoders.raster_seq import RasterSeqEncoder
from georsct.adapters.spatial_to_embedding import SpatialToEmbedding, SpatialPipeline
from georsct.loaders.raster import (
    make_synthetic_raster,
    make_synthetic_raster_pair,
    make_synthetic_raster_sequence,
)


IN_CHANNELS = 1
HIDDEN_DIM = 16
EMBEDDING_DIM = 64


class TestPatchConvPipeline:

    def test_single_raster_to_embedding(self):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        x = make_synthetic_raster(3, channels=IN_CHANNELS, seed=42)
        emb = pipe.encode(x)

        assert emb.shape == (3, EMBEDDING_DIM)
        assert emb.dtype == np.float64
        assert np.isfinite(emb).all()

    def test_batch_1(self):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        x = make_synthetic_raster(1, channels=IN_CHANNELS, seed=0)
        emb = pipe.encode(x)
        assert emb.shape == (1, EMBEDDING_DIM)

    def test_different_embedding_dims(self):
        enc = PatchConvEncoder(IN_CHANNELS, HIDDEN_DIM)
        x = make_synthetic_raster(2, channels=IN_CHANNELS, seed=0)
        for dim in [32, 64, 128]:
            adapter = SpatialToEmbedding(HIDDEN_DIM, dim)
            pipe = SpatialPipeline(enc, adapter)
            emb = pipe.encode(x)
            assert emb.shape == (2, dim)


class TestSiameseDiffPipeline:

    def test_pair_to_embedding(self):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        x, timestamps = make_synthetic_raster_pair(4, channels=IN_CHANNELS, seed=42)
        emb = pipe.encode(x, timestamps=timestamps)

        assert emb.shape == (4, EMBEDDING_DIM)
        assert emb.dtype == np.float64
        assert np.isfinite(emb).all()

    def test_change_produces_different_embedding(self):
        enc = SiameseDiffEncoder(IN_CHANNELS, HIDDEN_DIM)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        frame = make_synthetic_raster(1, channels=IN_CHANNELS, seed=0)
        no_change = torch.stack([frame, frame], dim=1)
        emb_same = pipe.encode(no_change)

        x_diff, _ = make_synthetic_raster_pair(1, channels=IN_CHANNELS, change_intensity=1.0, seed=0)
        emb_diff = pipe.encode(x_diff)

        assert not np.allclose(emb_same, emb_diff, atol=1e-3)


class TestRasterSeqPipeline:

    def test_sequence_to_embedding(self):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=64, patch_size=8)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        x, timestamps = make_synthetic_raster_sequence(2, n_frames=4, channels=IN_CHANNELS, seed=42)
        emb = pipe.encode(x, timestamps=timestamps)

        assert emb.shape == (2, EMBEDDING_DIM)
        assert emb.dtype == np.float64
        assert np.isfinite(emb).all()

    def test_long_sequence(self):
        enc = RasterSeqEncoder(IN_CHANNELS, HIDDEN_DIM, embed_dim=32, patch_size=16)
        adapter = SpatialToEmbedding(HIDDEN_DIM, EMBEDDING_DIM)
        pipe = SpatialPipeline(enc, adapter)

        x, t = make_synthetic_raster_sequence(1, n_frames=12, channels=IN_CHANNELS, height=32, width=32, seed=0)
        emb = pipe.encode(x, timestamps=t)

        assert emb.shape == (1, EMBEDDING_DIM)
        assert np.isfinite(emb).all()
