"""RSN simplex computation.

Pure functions -- no I/O, no S3, no SQL.
R (Relevance), S_sup (Superfluous), N (Noise) on the unit simplex.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RSNPoint:
    """A point on the RSN simplex (R + S_sup + N = 1)."""

    R: float
    S_sup: float
    N: float

    def __post_init__(self):
        total = self.R + self.S_sup + self.N
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RSN must sum to 1.0, got {total:.6f}")


def normalize_to_simplex(r_raw: float, s_raw: float, n_raw: float) -> RSNPoint:
    """Project raw scores onto the unit simplex."""
    total = r_raw + s_raw + n_raw
    if total < 1e-12:
        return RSNPoint(R=1 / 3, S_sup=1 / 3, N=1 / 3)
    return RSNPoint(R=r_raw / total, S_sup=s_raw / total, N=n_raw / total)


def simplex_distance(a: RSNPoint, b: RSNPoint) -> float:
    """Euclidean distance between two simplex points."""
    return float(np.sqrt((a.R - b.R) ** 2 + (a.S_sup - b.S_sup) ** 2 + (a.N - b.N) ** 2))
