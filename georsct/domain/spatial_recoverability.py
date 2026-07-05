"""Spatial recoverability of learned representations.

Delegates to rsct_spatial.topology.recoverability for the math.
Re-exports types and functions under the georsct namespace.
Keeps georsct-specific gate_3b_decision re-export and KAPPA_REGISTRY_ENTRY.
"""

# Re-export spatial math from upstream
from rsct_spatial.topology.recoverability import (  # noqa: F401
    gabriel_graph,
    count_crossings,
    mantel_correlation,
    mds_stress,
    coordinate_lift,
    SpatialRecoverabilityResult,
    compute_spatial_recoverability,
    AdversarialPermutationResult,
    adversarial_geography_permutation,
)


# =========================================================================
# Gate 3B decision -- georsct application layer (P4 separation)
# =========================================================================

def gate_3b_decision(
    forward_score: float,
    spatial_recoverability: float,
    forward_floor: float = 0.0,
    reconstruct_floor: float = 0.3,
) -> str:
    """Gate 3B: spatial recoverability.

    .. deprecated::
        Import from ``georsct.application.use_cases.gate_3b_decision``
        instead.
    """
    from georsct.application.use_cases.gate_3b_decision import (
        gate_3b_decision as _canonical,
    )
    return _canonical(
        forward_score, spatial_recoverability, forward_floor, reconstruct_floor,
    )


# =========================================================================
# KAPPA REGISTRY ENTRY (georsct domain metadata)
# =========================================================================

KAPPA_REGISTRY_ENTRY = {
    "name": "spatial_recoverability",
    "full_name": "Spatial Recoverability Score",
    "domain": "spatial_topology",
    "formula": "1 - (crossings - baseline) / max_possible_crossings",
    "inputs": ["embeddings (n, d)", "coords2d (n, 2)"],
    "range": "[0, 1]",
    "monotonicity": "higher = more planar-consistent implied topology",
    "gate": "Gate 3B (spatial recoverability)",
    "gate_condition": (
        "forward_score >= floor AND spatial_recoverability < reconstruct_floor "
        "=> RE_ENCODE"
    ),
    "orthogonality": (
        "spatial_randomness measures residual autocorrelation (Moran's I). "
        "spatial_recoverability measures planarity of the implied neighbor graph. "
        "Different objects: clustered errors (high spatial_randomness) can coexist "
        "with a perfectly planar representation, and vice versa."
    ),
    "lineage": (
        "S018U backward recoverability concept -> "
        "geospatial topology instantiation"
    ),
    "status": "EXPERIMENTAL",
    "corroboration": [
        "stress (MDS)",
        "mantel_r (distance correlation)",
        "coordinate_lift (location guard)",
    ],
    "admitted": "2026-06-11",
}
