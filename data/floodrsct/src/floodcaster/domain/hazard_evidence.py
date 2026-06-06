"""Hazard evidence computation -- flood fill, inundation masking.

Pure functions -- no I/O, no S3, no SQL.
Source pattern: LGAP Ch.9 B19730_09_03-flood.py (4-way non-recursive flood fill).
"""

import numpy as np


def flood_fill(
    col: int,
    row: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Non-recursive 4-way flood fill on a binary mask.

    Crawls from (col, row) through connected cells where mask == 1.
    Returns int8 array with 1 for inundated cells, 0 otherwise.

    Source: B19730_09_03-flood.py (Joel Lawhead, PyShp creator).

    Args:
        col: Starting column (x) in the mask.
        row: Starting row (y) in the mask.
        mask: 2D array of 0s and 1s (1 = below threshold).
    """
    filled: set[tuple[int, int]] = set()
    fill: set[tuple[int, int]] = {(col, row)}
    width = mask.shape[1] - 1
    height = mask.shape[0] - 1
    flood = np.zeros_like(mask, dtype=np.int8)

    while fill:
        x, y = fill.pop()
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if mask[y, x] == 1:
            flood[y, x] = 1
            filled.add((x, y))
            for nb in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if nb not in filled:
                    fill.add(nb)

    return flood


def mask_elevation_threshold(
    dem: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    """Create binary inundation mask from DEM and water level threshold.

    Returns 1 where elevation < threshold, 0 elsewhere.

    Args:
        dem: 2D elevation array (meters above sea level).
        threshold_m: Water surface elevation threshold.
    """
    return np.where(dem < threshold_m, 1, 0).astype(np.int8)


def compute_inundation_extent(
    dem: np.ndarray,
    threshold_m: float,
    seed_col: int,
    seed_row: int,
) -> np.ndarray:
    """Full inundation workflow: threshold DEM, then flood fill from seed.

    Combines mask_elevation_threshold + flood_fill into a single call.
    """
    mask = mask_elevation_threshold(dem, threshold_m)
    return flood_fill(seed_col, seed_row, mask)
