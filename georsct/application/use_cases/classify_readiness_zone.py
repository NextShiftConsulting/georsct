"""Use case: Classify readiness zone from QCT scores.

Maps quality/compatibility/turbulence to PublicDecision.
"""

from georsct.domain.certificate import PublicDecision


def classify(
    r2: float,
    silhouette: float,
    moran_p: float,
    kappa_geom: float,
    r_threshold: float = 0.3,
    s_threshold: float = 0.4,
    kappa_threshold: float = 0.7,
) -> PublicDecision:
    """Gate-based classification per CLAUDE.md quality gates.

    Stage 1: R >= 0.3 (relevance ~ model fit)
    Stage 2: S >= 0.4 (stability ~ clustering quality)
    Stage 3: moran_p — not yet wired (spatial autocorrelation check)
    Stage 4: kappa >= 0.7 (geometry compatibility)
    """
    # TODO: Stage 3 — moran_p is accepted but not yet used in decision logic.
    # Requires defining the residual Moran's I threshold (see ADR-048).
    if r2 < r_threshold:
        return PublicDecision.REFUSE
    if silhouette < s_threshold:
        return PublicDecision.CAUTION
    if kappa_geom < kappa_threshold:
        return PublicDecision.CAUTION
    return PublicDecision.EXECUTE
