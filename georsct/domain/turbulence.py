"""Turbulence scoring via spatial autocorrelation.

Delegates to rsct_spatial.weights.autocorrelation for the math.
Re-exports types and functions under the georsct namespace.
"""

# Re-export from upstream
from rsct_spatial.weights.autocorrelation import (  # noqa: F401
    TurbulenceResult,
    QUADRANT_LABELS,
    compute_global_autocorrelation,
    compute_lisa_clusters,
    score_turbulence,
    morans_i,
    compute_spatial_randomness,
)
