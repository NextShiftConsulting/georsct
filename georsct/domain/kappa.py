"""Kappa computation from spatial geometry.

Delegates to rsct_spatial.weights.builders for the math.
Re-exports types and functions under the georsct namespace.
"""

# Re-export from upstream
from rsct_spatial.weights.builders import (  # noqa: F401
    GeometryKappa,
    build_weights_queen,
    build_weights_knn,
    build_weights_from_adjacency,
    compute_spatial_connectivity,
    compute_support_coverage,
    compute_scale_stability,
    compute_geometry_kappa,
)
