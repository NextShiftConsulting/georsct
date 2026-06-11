"""Five-construct divergence matrix -- pairwise certificate distance.

Pure domain math.  No I/O, no S3, no pandas.

Computes Euclidean distance in (forward_score, kappa_spatial,
kappa_reconstruct) space between per-construct certificates.
All three axes are clamped to [0, 1] at certificate creation
(R2 can be negative; Moran's I can exceed |1|), so no rescaling is needed.

Euclidean over cosine: magnitude-sensitive (two constructs scoring
0.1 vs 0.9 IS different from 0.5 vs 0.6).

Euclidean over Mahalanobis: no cross-geography covariance available
(only ~5 scenarios).

P9: summarize_divergence() returns flat JSON for audit replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Sequence

import numpy as np

from georsct.domain.construct_certificate import (
    ConstructCertificate,
    ConstructLabel,
)


# ---------------------------------------------------------------------------
# Pairwise distance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairwiseDivergence:
    """Distance between two ConstructCertificates."""

    construct_a: ConstructLabel
    construct_b: ConstructLabel
    euclidean_distance: float
    forward_delta: float
    kappa_spatial_delta: float
    kappa_reconstruct_delta: float
    both_available: bool


def compute_certificate_distance(
    a: ConstructCertificate,
    b: ConstructCertificate,
) -> PairwiseDivergence:
    """Euclidean distance in (forward, kappa_spatial, kappa_reconstruct).

    Returns NaN distance when either certificate is missing.
    """
    both = a.target_available and b.target_available

    if not both:
        return PairwiseDivergence(
            construct_a=a.construct,
            construct_b=b.construct,
            euclidean_distance=float("nan"),
            forward_delta=float("nan"),
            kappa_spatial_delta=float("nan"),
            kappa_reconstruct_delta=float("nan"),
            both_available=False,
        )

    df = a.forward_score - b.forward_score
    ds = a.kappa_spatial - b.kappa_spatial
    dr = a.kappa_reconstruct - b.kappa_reconstruct

    dist = math.sqrt(df * df + ds * ds + dr * dr)

    return PairwiseDivergence(
        construct_a=a.construct,
        construct_b=b.construct,
        euclidean_distance=dist,
        forward_delta=df,
        kappa_spatial_delta=ds,
        kappa_reconstruct_delta=dr,
        both_available=True,
    )


# ---------------------------------------------------------------------------
# Divergence matrix
# ---------------------------------------------------------------------------

# Canonical ordering for the matrix rows/columns.
CONSTRUCT_ORDER: tuple[ConstructLabel, ...] = (
    ConstructLabel.JRC,
    ConstructLabel.DELTARES,
    ConstructLabel.FEMA,
    ConstructLabel.FAST,
    ConstructLabel.NFIP,
)


@dataclass(frozen=True)
class DivergenceMatrix:
    """5x5 pairwise certificate distance matrix.

    Measurement artifact for the SIGSPATIAL 2027 paper.
    """

    certificates: tuple[ConstructCertificate, ...]
    pairwise: tuple[PairwiseDivergence, ...]
    matrix: np.ndarray  # (5, 5) symmetric, zero diagonal
    n_available: int
    timestamp: str
    geography_id: str
    mean_distance: float
    max_distance: float
    max_pair: tuple[ConstructLabel, ConstructLabel]


def build_divergence_matrix(
    certificates: Sequence[ConstructCertificate],
    geography_id: str = "",
) -> DivergenceMatrix:
    """Assemble the 5x5 divergence matrix from up to 5 certificates.

    Args:
        certificates: One ConstructCertificate per construct.
            Missing constructs should use ConstructCertificate.missing().
            Must contain exactly one certificate per ConstructLabel.
        geography_id: Scenario or region label for provenance.

    Returns:
        DivergenceMatrix with symmetric distance matrix.

    Raises:
        ValueError: If not exactly 5 certificates or duplicate constructs.
    """
    if len(certificates) != len(CONSTRUCT_ORDER):
        raise ValueError(
            f"Expected {len(CONSTRUCT_ORDER)} certificates, got {len(certificates)}"
        )

    seen = set()
    for c in certificates:
        if c.construct in seen:
            raise ValueError(f"Duplicate construct: {c.construct}")
        seen.add(c.construct)

    # Re-order to canonical ordering
    cert_map = {c.construct: c for c in certificates}
    ordered = tuple(cert_map[label] for label in CONSTRUCT_ORDER)

    n = len(CONSTRUCT_ORDER)
    mat = np.zeros((n, n), dtype=float)
    pairs: list[PairwiseDivergence] = []

    for i, j in combinations(range(n), 2):
        pd_ = compute_certificate_distance(ordered[i], ordered[j])
        pairs.append(pd_)
        mat[i, j] = pd_.euclidean_distance
        mat[j, i] = pd_.euclidean_distance

    n_available = sum(1 for c in ordered if c.target_available)

    # Summary stats over finite distances only
    finite_dists = [p.euclidean_distance for p in pairs if math.isfinite(p.euclidean_distance)]
    if finite_dists:
        mean_d = float(np.mean(finite_dists))
        max_d = max(finite_dists)
        max_pair_obj = max(
            (p for p in pairs if math.isfinite(p.euclidean_distance)),
            key=lambda p: p.euclidean_distance,
        )
        max_pair_labels = (max_pair_obj.construct_a, max_pair_obj.construct_b)
    else:
        mean_d = float("nan")
        max_d = float("nan")
        max_pair_labels = (CONSTRUCT_ORDER[0], CONSTRUCT_ORDER[1])

    return DivergenceMatrix(
        certificates=ordered,
        pairwise=tuple(pairs),
        matrix=mat,
        n_available=n_available,
        timestamp=datetime.now(timezone.utc).isoformat(),
        geography_id=geography_id,
        mean_distance=mean_d,
        max_distance=max_d,
        max_pair=max_pair_labels,
    )


# ---------------------------------------------------------------------------
# Serialization (P9: replayability)
# ---------------------------------------------------------------------------

def summarize_divergence(dm: DivergenceMatrix) -> dict:
    """Flat dict for JSON serialization and audit logging.

    Every field is JSON-safe (no numpy types, no dataclass nesting).
    """
    def _safe(v: float) -> object:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    per_construct = []
    for c in dm.certificates:
        per_construct.append({
            "construct": c.construct.value,
            "target_column": c.target_column,
            "target_available": c.target_available,
            "forward_score": _safe(c.forward_score),
            "kappa_spatial": _safe(c.kappa_spatial),
            "kappa_reconstruct": _safe(c.kappa_reconstruct),
            "morans_i": _safe(c.morans_i),
            "n_regions": c.n_regions,
            "n_observations": c.n_observations,
            "n_finite_targets": c.n_finite_targets,
            "kappa_reconstruct_source": c.kappa_reconstruct_source,
            "kappa_spatial_source": c.kappa_spatial_source,
            "warnings": list(c.warnings),
        })

    pairwise = []
    for p in dm.pairwise:
        pairwise.append({
            "construct_a": p.construct_a.value,
            "construct_b": p.construct_b.value,
            "euclidean_distance": _safe(p.euclidean_distance),
            "forward_delta": _safe(p.forward_delta),
            "kappa_spatial_delta": _safe(p.kappa_spatial_delta),
            "kappa_reconstruct_delta": _safe(p.kappa_reconstruct_delta),
            "both_available": p.both_available,
        })

    # Distance matrix as nested list (JSON-safe)
    mat_list = []
    for row in dm.matrix:
        mat_list.append([_safe(float(v)) for v in row])

    return {
        "geography_id": dm.geography_id,
        "timestamp": dm.timestamp,
        "n_available": dm.n_available,
        "construct_order": [c.value for c in CONSTRUCT_ORDER],
        "mean_distance": _safe(dm.mean_distance),
        "max_distance": _safe(dm.max_distance),
        "max_pair": [dm.max_pair[0].value, dm.max_pair[1].value],
        "per_construct": per_construct,
        "pairwise": pairwise,
        "distance_matrix": mat_list,
    }
