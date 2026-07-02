"""Raster loading utilities: numpy/GeoTIFF -> torch tensor.

Converts geospatial rasters to PyTorch tensors suitable for
spatial encoder consumption. Handles normalization, cropping,
and synthetic data generation for testing.
"""

import numpy as np
import torch
from numpy.typing import NDArray


def numpy_to_tensor(
    arr: NDArray,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a numpy raster array to a torch tensor.

    Args:
        arr: Numpy array with shape (H, W), (C, H, W), or (T, C, H, W).
        normalize: If True, normalize to [0, 1] range per-channel.

    Returns:
        Float32 torch tensor with same shape.
    """
    tensor = torch.from_numpy(arr.astype(np.float32))

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if normalize and tensor.numel() > 0:
        if tensor.ndim == 3:
            for c in range(tensor.shape[0]):
                ch = tensor[c]
                vmin, vmax = ch.min(), ch.max()
                if vmax > vmin:
                    tensor[c] = (ch - vmin) / (vmax - vmin)
        elif tensor.ndim == 4:
            for t in range(tensor.shape[0]):
                for c in range(tensor.shape[1]):
                    ch = tensor[t, c]
                    vmin, vmax = ch.min(), ch.max()
                    if vmax > vmin:
                        tensor[t, c] = (ch - vmin) / (vmax - vmin)

    return tensor


def center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop a spatial tensor to (size, size).

    Args:
        tensor: Tensor with spatial dims as last two dimensions.
        size: Target height and width.

    Returns:
        Cropped tensor.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h < size or w < size:
        raise ValueError(
            f"Tensor spatial dims ({h}, {w}) smaller than crop size {size}"
        )
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return tensor[..., y0:y0 + size, x0:x0 + size]


def make_synthetic_raster(
    batch: int,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate synthetic raster data for testing.

    Creates smooth spatial patterns (not random noise) to simulate
    realistic geospatial data with spatial autocorrelation.

    Returns:
        (batch, channels, height, width) float32 tensor in [0, 1].
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.zeros(batch, channels, height, width)
    yy = torch.linspace(-1, 1, height)
    xx = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    for b in range(batch):
        for c in range(channels):
            freq = 1.0 + torch.rand(1).item() * 3.0
            phase = torch.rand(1).item() * 6.28
            pattern = torch.sin(freq * grid_x + phase) * torch.cos(freq * grid_y + phase)
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            x[b, c] = pattern + 0.05 * torch.randn(height, width)
            x[b, c] = x[b, c].clamp(0, 1)

    return x


def make_synthetic_raster_pair(
    batch: int,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    change_intensity: float = 0.3,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic before/after raster pairs for change detection.

    Returns:
        (x_pair, timestamps) where x_pair is (batch, 2, C, H, W) and
        timestamps is (batch, 2) with before/after times.
    """
    if seed is not None:
        torch.manual_seed(seed)

    before = make_synthetic_raster(batch, channels, height, width)

    after = before.clone()
    for b in range(batch):
        cy, cx = height // 2, width // 2
        r = max(4, int(height * 0.3))
        y0, y1 = max(0, cy - r), min(height, cy + r)
        x0, x1 = max(0, cx - r), min(width, cx + r)
        after[b, :, y0:y1, x0:x1] += change_intensity * torch.randn(channels, y1 - y0, x1 - x0)
        after[b] = after[b].clamp(0, 1)

    x_pair = torch.stack([before, after], dim=1)
    timestamps = torch.tensor([[0.0, 24.0]] * batch)

    return x_pair, timestamps


def make_synthetic_raster_sequence(
    batch: int,
    n_frames: int = 8,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic temporal raster sequence.

    Simulates a moving precipitation pattern across frames.

    Returns:
        (x_seq, timestamps) where x_seq is (batch, T, C, H, W) and
        timestamps is (batch, T) with irregular spacing.
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.zeros(batch, n_frames, channels, height, width)
    yy = torch.linspace(-1, 1, height)
    xx = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    for b in range(batch):
        cx_start = -0.5 + torch.rand(1).item()
        cy_start = -0.5 + torch.rand(1).item()
        dx = (0.5 + torch.rand(1).item()) / n_frames
        dy = (0.3 + torch.rand(1).item() * 0.4) / n_frames

        for t in range(n_frames):
            cx = cx_start + dx * t
            cy = cy_start + dy * t
            r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            storm = torch.exp(-r2 / 0.1) * (0.5 + 0.5 * torch.rand(1).item())
            for c in range(channels):
                x[b, t, c] = storm + 0.02 * torch.randn(height, width)
                x[b, t, c] = x[b, t, c].clamp(0, 1)

    timestamps = torch.sort(torch.rand(batch, n_frames), dim=1)[0] * 48.0

    return x, timestamps
