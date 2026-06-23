"""R3-MAE: Masked Tabular Autoencoder — non-JEPA learned control arm.

Matched-capacity masked autoencoder that reconstructs raw feature values
from visible context. This is the control for R3-TJEPA: same architecture,
same mask ratio, same capacity — but reconstruction loss in feature space
instead of latent prediction.

The comparison R3-TJEPA vs R3-MAE isolates the JEPA-specific effect (latent
prediction + EMA target encoder) from the generic learned-encoder effect
(any neural network finding feature interactions).

Key differences from TJEPA:
  1. Loss is MSE in FEATURE space (not cosine in latent space)
  2. Has a DECODER (MLP) instead of a predictor+target encoder
  3. No EMA target encoder — single encoder, trained end-to-end
  4. Prediction target is raw masked feature values, not latent representations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MAEConfig:
    """Configuration for Masked Tabular AE. Mirrors TJEPAConfig for capacity match."""

    n_features: int
    embed_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    mask_ratio: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    batch_size: int = 256
    seed: int = 42


class MaskedTabularAE(nn.Module):
    """Masked Tabular Autoencoder: reconstructs raw features from visible context.

    Architecture:
        Input features -> FeatureTokenizer -> per-feature tokens
        Context tokens (visible) -> Encoder -> context representations
        Context repr -> Decoder -> reconstructed feature values
        Loss = MSE(reconstructed, original) at masked positions
    """

    def __init__(self, config: MAEConfig):
        super().__init__()
        self.config = config

        # Reuse same tokenizer and encoder as TJEPA for capacity match
        from georsct.encoders.tjepa import FeatureTokenizer, TransformerEncoder

        self.tokenizer = FeatureTokenizer(config.n_features, config.embed_dim)
        self.encoder = TransformerEncoder(
            config.embed_dim, config.hidden_dim, config.n_layers,
        )

        # Decoder: maps encoder output back to feature space
        # Two-layer MLP per the DOE spec
        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),  # predict one scalar per feature token
        )

        self._fitted = False
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _generate_mask(self, batch_size: int, rng: torch.Generator) -> torch.Tensor:
        """Generate random feature mask. True = visible, False = masked."""
        n_mask = max(1, int(self.config.n_features * self.config.mask_ratio))
        n_mask = min(n_mask, self.config.n_features - 1)

        mask = torch.ones(batch_size, self.config.n_features, dtype=torch.bool)
        for i in range(batch_size):
            idx = torch.randperm(self.config.n_features, generator=rng)[:n_mask]
            mask[i, idx] = False
        return mask

    def forward_loss(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """Compute reconstruction loss at masked positions."""
        batch_size = x.shape[0]

        all_tokens = self.tokenizer(x)

        mask = self._generate_mask(batch_size, rng).to(x.device)

        # Encode with mask (zero out masked tokens)
        encoded = self.encoder(all_tokens, mask=mask)

        # Decode: predict raw feature values
        decoded = self.decoder(encoded).squeeze(-1)  # (batch, n_features)

        # Loss: MSE at masked positions only (reconstruct raw values)
        target_mask = ~mask
        if not target_mask.any():
            return torch.tensor(0.0, device=x.device)

        pred_masked = decoded[target_mask]
        true_masked = x[target_mask]

        return F.mse_loss(pred_masked, true_masked)

    def fit(self, X: np.ndarray) -> MaskedTabularAE:
        """Train the autoencoder on tabular features.

        Args:
            X: (n_samples, n_features) feature matrix.

        Returns:
            self (for chaining).
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = torch.Generator().manual_seed(cfg.seed)

        X = X.copy().astype(np.float32)
        self._feature_mean = np.nanmean(X, axis=0)
        self._feature_std = np.nanstd(X, axis=0)
        self._feature_std[self._feature_std < 1e-8] = 1.0
        X = np.where(np.isnan(X), self._feature_mean, X)
        X = (X - self._feature_mean) / self._feature_std

        X_t = torch.from_numpy(X)
        n = len(X_t)

        optimizer = torch.optim.AdamW(
            self.parameters(),
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

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg = epoch_loss / max(n_batches, 1)
                log.info("MAE epoch %d/%d  loss=%.4f", epoch + 1, cfg.n_epochs, avg)

        self._fitted = True
        self.eval()
        return self

    @torch.no_grad()
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from the encoder (no masking at inference).

        Args:
            X: (n_samples, n_features) feature matrix.

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

        tokens = self.tokenizer(X_t)
        encoded = self.encoder(tokens)
        embeddings = encoded.mean(dim=1)  # (n_samples, embed_dim)

        return embeddings.numpy()

    def param_count(self) -> int:
        """Total trainable parameters."""
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
        log.info("Saved MAE to %s", path)

    @classmethod
    def load(cls, path: str) -> MaskedTabularAE:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model._feature_mean = ckpt["feature_mean"]
        model._feature_std = ckpt["feature_std"]
        model._fitted = ckpt["fitted"]
        model.eval()
        return model
