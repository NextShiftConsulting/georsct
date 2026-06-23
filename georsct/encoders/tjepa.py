"""R3-TJEPA: Masked Tabular JEPA encoder for GeoRSCT.

JEPA-style masked latent prediction over spatial tabular features.
Predicts latent representations of masked feature subsets from visible
context -- NOT raw feature reconstruction.

Key JEPA discipline (avoiding "just a masked autoencoder" trap):
  1. Predicts LATENT targets (target encoder output), not raw values
  2. Uses EMA target encoder (not shared weights, not independent)
  3. Loss is in representation space (cosine + L2), not feature space
  4. No decoder / no pixel-level reconstruction objective

Reference: Assran et al. (2023) "Self-Supervised Learning from Images
with a Joint-Embedding Predictive Architecture" (CVPR 2023).

Usage:
    encoder = TabularJEPA(n_features=31, embed_dim=128)
    encoder.fit(X, n_epochs=100, mask_ratio=0.3)
    embeddings = encoder.encode(X)  # (n_samples, embed_dim)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TJEPAConfig:
    """Configuration for Tabular JEPA."""

    n_features: int
    embed_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    mask_ratio: float = 0.3
    ema_decay: float = 0.996
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    batch_size: int = 256
    seed: int = 42
    n_mask_seeds: int = 10  # for stability measurement


class FeatureTokenizer(nn.Module):
    """Project each scalar feature to a d-dim token embedding."""

    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        # Per-feature linear projection (no shared weights across features)
        self.projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) -> (batch, n_features, embed_dim)."""
        tokens = []
        for i, proj in enumerate(self.projections):
            tokens.append(proj(x[:, i : i + 1]))  # (batch, embed_dim)
        return torch.stack(tokens, dim=1)  # (batch, n_features, embed_dim)


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for tabular tokens."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=max(1, embed_dim // 64),
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (batch, seq, embed_dim), mask: (batch, seq) bool True=keep."""
        if mask is not None:
            # Apply mask: zero out masked positions before encoding
            tokens = tokens * mask.unsqueeze(-1).float()
        out = self.encoder(tokens)
        return self.norm(out)


class Predictor(nn.Module):
    """Predicts target token embeddings from context encoder output."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(context_tokens)


class TabularJEPA(nn.Module):
    """Tabular JEPA: masked latent prediction over feature columns.

    Architecture:
        Input features → FeatureTokenizer → per-feature tokens
        Context tokens (visible) → Context Encoder → context representations
        Context repr → Predictor → predicted target representations
        Target tokens (masked) → Target Encoder (EMA) → target representations
        Loss = cosine distance(predicted, target) in latent space
    """

    def __init__(self, config: TJEPAConfig):
        super().__init__()
        self.config = config

        # Feature tokenizer (shared between context and target paths)
        self.tokenizer = FeatureTokenizer(config.n_features, config.embed_dim)

        # Context encoder (trained by gradient descent)
        self.context_encoder = TransformerEncoder(
            config.embed_dim, config.hidden_dim, config.n_layers,
        )

        # Target encoder (EMA copy of context encoder -- NOT trained by gradients)
        self.target_encoder = TransformerEncoder(
            config.embed_dim, config.hidden_dim, config.n_layers,
        )
        # Initialize target encoder as exact copy
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Freeze target encoder -- only updated via EMA
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: maps context representations to target representation space
        self.predictor = Predictor(config.embed_dim, config.hidden_dim)

        self._fitted = False
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Exponential moving average update of target encoder."""
        decay = self.config.ema_decay
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(decay).add_(p_ctx.data, alpha=1.0 - decay)

    def _generate_mask(self, batch_size: int, rng: torch.Generator) -> torch.Tensor:
        """Generate random feature mask. True = visible (context), False = masked (target)."""
        n_mask = max(1, int(self.config.n_features * self.config.mask_ratio))
        n_mask = min(n_mask, self.config.n_features - 1)  # keep at least 1 visible

        mask = torch.ones(batch_size, self.config.n_features, dtype=torch.bool)
        for i in range(batch_size):
            idx = torch.randperm(self.config.n_features, generator=rng)[:n_mask]
            mask[i, idx] = False
        return mask

    def forward_loss(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """Compute JEPA loss: predict target embeddings from context.

        Returns scalar loss (cosine distance in latent space).
        """
        batch_size = x.shape[0]

        # Tokenize all features
        all_tokens = self.tokenizer(x)  # (batch, n_features, embed_dim)

        # Generate mask
        mask = self._generate_mask(batch_size, rng)
        device = x.device
        mask = mask.to(device)

        # Context path: encode visible tokens
        context_repr = self.context_encoder(all_tokens, mask=mask)

        # Predict target representations from context (at masked positions)
        predicted = self.predictor(context_repr)  # (batch, n_features, embed_dim)

        # Target path: encode ALL tokens with EMA encoder (no mask)
        with torch.no_grad():
            target_repr = self.target_encoder(all_tokens)  # (batch, n_features, embed_dim)

        # Loss: cosine distance at masked positions only
        target_mask = ~mask  # True = masked positions (prediction targets)
        if not target_mask.any():
            return torch.tensor(0.0, device=device)

        # Gather masked position representations
        pred_masked = predicted[target_mask]   # (n_masked_total, embed_dim)
        tgt_masked = target_repr[target_mask]  # (n_masked_total, embed_dim)

        # Cosine similarity loss (1 - cosine_sim) + small L2 regularization
        cosine_loss = 1.0 - F.cosine_similarity(pred_masked, tgt_masked, dim=-1).mean()
        l2_loss = F.mse_loss(pred_masked, tgt_masked)

        return cosine_loss + 0.1 * l2_loss

    def fit(self, X: np.ndarray) -> TabularJEPA:
        """Train the JEPA encoder on tabular features.

        Args:
            X: (n_samples, n_features) feature matrix. NaN values are
               imputed with feature means before training.

        Returns:
            self (for chaining).
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = torch.Generator().manual_seed(cfg.seed)

        # Standardize (store for encode-time)
        X = X.copy().astype(np.float32)
        self._feature_mean = np.nanmean(X, axis=0)
        self._feature_std = np.nanstd(X, axis=0)
        self._feature_std[self._feature_std < 1e-8] = 1.0
        X = np.where(np.isnan(X), self._feature_mean, X)
        X = (X - self._feature_mean) / self._feature_std

        X_t = torch.from_numpy(X)
        n = len(X_t)

        optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.tokenizer.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.train()
        for epoch in range(cfg.n_epochs):
            perm = torch.randperm(n, generator=rng)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, cfg.batch_size):
                batch = X_t[perm[start : start + cfg.batch_size]]
                if len(batch) < 2:
                    continue

                loss = self.forward_loss(batch, rng)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                # EMA update of target encoder (after each optimizer step)
                self._ema_update()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg = epoch_loss / max(n_batches, 1)
                log.info("TJEPA epoch %d/%d  loss=%.4f", epoch + 1, cfg.n_epochs, avg)

        self._fitted = True
        self.eval()
        return self

    @torch.no_grad()
    def encode(self, X: np.ndarray, mask_seed: int | None = None) -> np.ndarray:
        """Extract embeddings using the target encoder (EMA).

        Args:
            X: (n_samples, n_features) feature matrix.
            mask_seed: If None, encode all features (no masking).
                If set, apply deterministic mask for stability testing.

        Returns:
            (n_samples, embed_dim) embeddings.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")

        self.eval()
        X = X.copy().astype(np.float32)
        X = np.where(np.isnan(X), self._feature_mean, X)
        X = (X - self._feature_mean) / self._feature_std
        X_t = torch.from_numpy(X)

        # Tokenize
        tokens = self.tokenizer(X_t)

        if mask_seed is not None:
            rng = torch.Generator().manual_seed(mask_seed)
            mask = self._generate_mask(len(X_t), rng)
        else:
            mask = None

        # Use TARGET encoder (EMA) for final embeddings -- this is the
        # JEPA convention: the target encoder produces the canonical representation
        repr_out = self.target_encoder(tokens, mask=mask)

        # Pool: mean over feature tokens → single vector per sample
        embeddings = repr_out.mean(dim=1)  # (n_samples, embed_dim)

        return embeddings.numpy()

    @torch.no_grad()
    def encode_multi_mask(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode with multiple mask seeds for stability measurement.

        Returns:
            mean_embedding: (n_samples, embed_dim) averaged across masks
            std_embedding: (n_samples, embed_dim) std across masks
        """
        all_embeddings = []
        for seed in range(self.config.n_mask_seeds):
            emb = self.encode(X, mask_seed=seed + 1000)
            all_embeddings.append(emb)

        stacked = np.stack(all_embeddings, axis=0)  # (n_seeds, n_samples, embed_dim)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def param_count(self) -> int:
        """Total trainable parameters (context encoder + predictor + tokenizer)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model weights and normalization stats."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "fitted": self._fitted,
        }, path)
        log.info("Saved TJEPA to %s", path)

    @classmethod
    def load(cls, path: str) -> TabularJEPA:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model._feature_mean = ckpt["feature_mean"]
        model._feature_std = ckpt["feature_std"]
        model._fitted = ckpt["fitted"]
        model.eval()
        return model
