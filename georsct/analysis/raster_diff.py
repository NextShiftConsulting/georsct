"""Raster differencing analysis tool.

Diagnostic module for computing and analyzing frame differences
in temporal raster sequences. Not part of the encoding pipeline.

Use cases:
    - Visualize change magnitude between frames
    - Compute temporal profiles at specific pixels
    - Identify regions of maximum change
    - Validate that encoder captures meaningful spatial variation
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class DiffResult:
    """Result from frame differencing analysis."""
    mean_abs_diff: float
    max_abs_diff: float
    change_fraction: float
    diff_map: NDArray


def compute_frame_diff(
    frame_a: NDArray,
    frame_b: NDArray,
    threshold: float = 0.1,
) -> DiffResult:
    """Compute difference between two raster frames.

    Args:
        frame_a: First frame (H, W) or (C, H, W).
        frame_b: Second frame, same shape as frame_a.
        threshold: Change detection threshold for change_fraction.

    Returns:
        DiffResult with statistics and spatial difference map.
    """
    diff = np.abs(frame_b.astype(np.float64) - frame_a.astype(np.float64))

    if diff.ndim == 3:
        diff_map = diff.mean(axis=0)
    else:
        diff_map = diff

    return DiffResult(
        mean_abs_diff=float(np.mean(diff_map)),
        max_abs_diff=float(np.max(diff_map)),
        change_fraction=float(np.mean(diff_map > threshold)),
        diff_map=diff_map,
    )


def temporal_profile(
    sequence: NDArray,
    row: int,
    col: int,
    channel: int = 0,
) -> NDArray:
    """Extract temporal profile at a specific pixel location.

    Args:
        sequence: Raster sequence (T, C, H, W) or (T, H, W).
        row: Pixel row index.
        col: Pixel column index.
        channel: Channel index (ignored if sequence is 3D).

    Returns:
        (T,) array of values at the specified pixel across time.
    """
    if sequence.ndim == 4:
        return sequence[:, channel, row, col].copy()
    return sequence[:, row, col].copy()


def change_magnitude_map(sequence: NDArray) -> NDArray:
    """Compute total change magnitude across a temporal sequence.

    Args:
        sequence: Raster sequence (T, H, W) or (T, C, H, W).

    Returns:
        (H, W) total change magnitude map.
    """
    if sequence.ndim == 4:
        seq = sequence.mean(axis=1)
    else:
        seq = sequence

    total = np.zeros_like(seq[0], dtype=np.float64)
    for t in range(1, len(seq)):
        total += np.abs(seq[t].astype(np.float64) - seq[t - 1].astype(np.float64))

    return total


def find_hotspots(
    change_map: NDArray,
    n_hotspots: int = 5,
) -> list[tuple[int, int, float]]:
    """Find pixels with highest total change.

    Args:
        change_map: (H, W) change magnitude map.
        n_hotspots: Number of top locations to return.

    Returns:
        List of (row, col, magnitude) tuples, sorted by magnitude descending.
    """
    flat = change_map.ravel()
    top_indices = np.argsort(flat)[-n_hotspots:][::-1]
    h, w = change_map.shape
    results = []
    for idx in top_indices:
        row, col = divmod(int(idx), w)
        results.append((row, col, float(change_map[row, col])))
    return results
