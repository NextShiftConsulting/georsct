"""Use case: Score quality, compatibility, and turbulence for a spatial domain.

Orchestrates domain functions through ports. No direct I/O.
"""

from georsct.domain.certificate import PublicDecision, ReadinessCertificate
from georsct.domain.rsn_simplex import RSNPoint, normalize_to_simplex
from georsct.ports.spatial_metric_source import SpatialMetricSource


def score_qct(
    metric_source: SpatialMetricSource,
    unit_ids: list[str],
    target_variable: str,
    feature_variables: list[str],
    w: "libpysal.weights.W",
) -> dict:
    """Compute quality, compatibility, turbulence for a set of spatial units.

    Orchestration only -- delegates to domain functions.
    Returns dict with keys: quality, compatibility, turbulence, rsn_point.
    """
    from georsct.domain import quality, compatibility, turbulence

    # Quality: spatial regression diagnostics
    y = metric_source.get_values(target_variable, unit_ids)
    X_cols = [metric_source.get_values(f, unit_ids) for f in feature_variables]
    import numpy as np
    X = np.column_stack(X_cols)
    coords = metric_source.get_coordinates(unit_ids)

    ols_diag = quality.fit_ols(y, X, feature_variables)

    # Turbulence: spatial autocorrelation of residuals
    turb = turbulence.score_turbulence(y, w)

    # Compatibility: spatially-constrained clustering quality
    compat = compatibility.cluster_spatially_constrained(X, w, n_clusters=5)

    return {
        "quality": ols_diag,
        "compatibility": compat,
        "turbulence": turb,
    }
