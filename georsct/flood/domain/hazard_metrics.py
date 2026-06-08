"""RSCT-specific composite scoring and explainability for flood hazard.

Pure functions -- no I/O, no S3, no SQL.

General-purpose flood hazard metrics (crest margin, rate of rise, NWS
severity, depth above ground, freeboard, discharge percentile, encounter
probability) live in floodcaster.hazard. This module contains only the
RSCT governance layer: the composite scoring formula and its decomposition,
which together form the audit artifact for explainability (P9, P11).

Re-exports floodcaster.hazard types for convenience so that consumers
importing from this module get the full hazard vocabulary.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Re-export general flood hazard types from floodcaster
from floodcaster.hazard import (  # noqa: F401
    GageThresholds,
    NWSSeverity,
    TemporalBasis,
    crest_margin,
    depth_above_ground,
    discharge_percentile,
    encounter_probability,
    freeboard,
    nws_severity,
    rate_of_rise,
    rate_of_rise_series,
)


# ---------------------------------------------------------------------------
# RSCT composite scoring (the one formula that is genuinely ours)
# ---------------------------------------------------------------------------

def composite_score(
    features: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted normalized composite risk score.

    Formula:
        S = sum(w_i * normalize(feature_i))

    This is the one formula that is genuinely ours. The weights and
    normalization are documented so every score decomposes into its
    contributing terms. That decomposition IS the explainability story
    and the audit artifact -- it's the part RSCT-style governance cares
    about, not the hydraulics.

    Features must be pre-normalized to [0, 1] before calling this
    function. Use normalize_minmax() for the normalization step.

    Args:
        features: 1D array of normalized feature values, each in [0, 1].
        weights: 1D array of weights (must sum to 1.0).

    Returns:
        Composite score in [0, 1].

    Raises:
        ValueError: If arrays have different lengths or weights don't sum to ~1.
    """
    if len(features) != len(weights):
        raise ValueError(
            f"features ({len(features)}) and weights ({len(weights)}) "
            f"must have the same length"
        )
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0, got {weight_sum:.8f}"
        )
    return float(np.dot(weights, features))


def normalize_minmax(
    values: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Min-max normalize to [0, 1].

    Args:
        values: 1D array of raw values.
        vmin: Floor (default: array min). Values below are clipped to 0.
        vmax: Ceiling (default: array max). Values above are clipped to 1.

    Returns:
        Normalized array in [0, 1]. Returns zeros if vmin == vmax.
    """
    lo = vmin if vmin is not None else float(np.nanmin(values))
    hi = vmax if vmax is not None else float(np.nanmax(values))
    if hi == lo:
        return np.zeros_like(values, dtype=np.float64)
    normed = (values - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0)


def score_decomposition(
    feature_names: Sequence[str],
    features: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, float | str]]:
    """Decompose a composite score into per-feature contributions.

    Returns a list of dicts, one per feature, with the feature name,
    normalized value, weight, and contribution (weight * value).
    Sorted by contribution descending.

    This decomposition is the explainability audit artifact -- every
    score can be traced back to its contributing terms.

    Args:
        feature_names: Names for each feature dimension.
        features: Normalized feature values.
        weights: Feature weights.

    Returns:
        List of {"feature", "value", "weight", "contribution"} dicts.
    """
    contributions = []
    for name, val, w in zip(feature_names, features, weights):
        contributions.append({
            "feature": name,
            "value": float(val),
            "weight": float(w),
            "contribution": float(w * val),
        })
    contributions.sort(key=lambda c: c["contribution"], reverse=True)
    return contributions
