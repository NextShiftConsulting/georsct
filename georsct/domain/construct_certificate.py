"""Per-construct certification for five-construct divergence.

Pure domain objects and computation -- no I/O, no S3, no pandas.

Each of five flood constructs (JRC, Deltares, FEMA, NFIP, FAST) encodes
a different aspect of flood risk.  This module defines the typed construct
labels, the per-construct certificate, and a pure Moran's I implementation
for spatial_randomness.

ADR-020 D8: every kappa value carries provenance.
ADR-034: typed enums for construct identity.
P4: measurement layer only -- no decisions emitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Construct identity (ADR-034: typed enum, not bare strings)
# ---------------------------------------------------------------------------

class ConstructLabel(str, Enum):
    """The five flood constructs certified independently.

    Each encodes a different measurement of flood risk:
      JRC       -- satellite-observed historical water presence
      DELTARES  -- physics-modeled inundation depth at return period
      FEMA      -- regulatory flood zone designation
      NFIP      -- administrative insurance loss per disaster
      FAST      -- engineering-modeled structural damage (Hazus)
    """

    JRC = "jrc_observed_water"
    DELTARES = "deltares_rp_depth"
    FEMA = "fema_regulatory_zone"
    NFIP = "nfip_administrative_loss"
    FAST = "fast_modeled_damage"


# Canonical target column for each construct.
# Used by the application use case; stored here because
# the mapping is part of the domain definition.
CONSTRUCT_TARGET_COLUMNS: dict[ConstructLabel, str] = {
    ConstructLabel.JRC: "jrc_occurrence_mean",
    ConstructLabel.DELTARES: "deltares_depth_ft_rp100",
    ConstructLabel.FEMA: "flood_pct_zone_a",
    ConstructLabel.NFIP: "obs_nfip_event_claims",
    ConstructLabel.FAST: "fast_total_loss_usd",
}

# Regression task for all constructs (continuous targets).
CONSTRUCT_TASK_TYPES: dict[ConstructLabel, str] = {
    ConstructLabel.JRC: "regression",
    ConstructLabel.DELTARES: "regression",
    ConstructLabel.FEMA: "regression",
    ConstructLabel.NFIP: "regression",
    ConstructLabel.FAST: "regression",
}


# ---------------------------------------------------------------------------
# spatial_randomness -- pure Moran's I (no pysal dependency)
# ---------------------------------------------------------------------------

def _morans_i(
    x: np.ndarray,
    W: sparse.csr_matrix,
) -> float:
    """Global Moran's I from a region-level vector and row-normalized W.

    Args:
        x: (n_regions,) numeric values.  NaN-safe: only finite entries used.
        W: (n_regions, n_regions) row-normalized spatial weights (CSR).

    Returns:
        Moran's I statistic.  NaN if fewer than 3 finite values
        or if denominator is zero.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return float("nan")

    Wm = W[mask][:, mask].tocsr()
    xm = x[mask]
    xm = xm - xm.mean()

    denom = float(np.dot(xm, xm))
    s0 = float(Wm.sum())
    if denom <= 0.0 or s0 <= 0.0:
        return float("nan")

    n = len(xm)
    numer = float(xm @ (Wm @ xm))
    return (n / s0) * (numer / denom)


def compute_spatial_randomness(
    residuals_by_region: np.ndarray,
    W_geo: sparse.csr_matrix,
) -> tuple[float, float]:
    """Compute spatial_randomness = 1 - |Moran's I| from region-level residuals.

    ADR-020 D8: returns (NaN, NaN) when data insufficient, never (0.0, 0.0).

    Args:
        residuals_by_region: (n_regions,) mean residual per region.
        W_geo: (n_regions, n_regions) row-normalized adjacency.

    Returns:
        (spatial_randomness, morans_i_raw)
    """
    I = _morans_i(residuals_by_region, W_geo)
    if not np.isfinite(I):
        return float("nan"), float("nan")
    kappa = float(np.clip(1.0 - abs(I), 0.0, 1.0))
    return kappa, float(I)


# ---------------------------------------------------------------------------
# Per-construct certificate (ADR-020 D8 provenance)
# ---------------------------------------------------------------------------

# Default provenance for spatial_recoverability.
_KR_PROVENANCE = {
    "kappa_source": "georsct.domain.spatial_recoverability",
    "kappa_formula": "1 - excess_crossings / max_possible",
    "kappa_authority": "RSCT-P008",
    "kappa_inputs": ("embeddings", "coords2d"),
}

# Default provenance for spatial_randomness.
_KS_PROVENANCE = {
    "kappa_source": "georsct.domain.construct_certificate",
    "kappa_formula": "1 - |Moran's I|",
    "kappa_authority": "RSCT-P008",
    "kappa_inputs": ("residuals_by_region", "W_geo"),
}


@dataclass(frozen=True)
class ConstructCertificate:
    """Certificate from certifying one geography under one construct.

    Measurement-layer artifact (P4).  No decisions emitted.

    ADR-020 D8 provenance fields are mandatory on valid certificates
    and explicitly None on missing certificates.
    """

    construct: ConstructLabel
    target_column: str

    # Scores -- all [0, 1] or NaN (never 0.0 for missing, per ADR-020 D8)
    forward_score: float
    spatial_randomness: float
    spatial_recoverability: float
    morans_i: float

    # Counts
    n_regions: int
    n_observations: int
    n_finite_targets: int

    # Availability
    target_available: bool

    # ADR-020 D8: kappa provenance (None on missing certificates)
    spatial_recoverability_source: Optional[str] = None
    spatial_recoverability_formula: Optional[str] = None
    spatial_recoverability_authority: Optional[str] = None
    spatial_recoverability_inputs: Optional[tuple[str, ...]] = None

    spatial_randomness_source: Optional[str] = None
    spatial_randomness_formula: Optional[str] = None
    spatial_randomness_authority: Optional[str] = None
    spatial_randomness_inputs: Optional[tuple[str, ...]] = None

    warnings: tuple[str, ...] = ()

    @classmethod
    def from_scores(
        cls,
        construct: ConstructLabel,
        target_column: str,
        forward_score: float,
        spatial_randomness: float,
        spatial_recoverability: float,
        morans_i: float,
        n_regions: int,
        n_observations: int,
        n_finite_targets: int,
        warnings: tuple[str, ...] = (),
    ) -> ConstructCertificate:
        """Build a valid certificate with default provenance."""
        return cls(
            construct=construct,
            target_column=target_column,
            forward_score=forward_score,
            spatial_randomness=spatial_randomness,
            spatial_recoverability=spatial_recoverability,
            morans_i=morans_i,
            n_regions=n_regions,
            n_observations=n_observations,
            n_finite_targets=n_finite_targets,
            target_available=True,
            spatial_recoverability_source=_KR_PROVENANCE["kappa_source"],
            spatial_recoverability_formula=_KR_PROVENANCE["kappa_formula"],
            spatial_recoverability_authority=_KR_PROVENANCE["kappa_authority"],
            spatial_recoverability_inputs=_KR_PROVENANCE["kappa_inputs"],
            spatial_randomness_source=_KS_PROVENANCE["kappa_source"],
            spatial_randomness_formula=_KS_PROVENANCE["kappa_formula"],
            spatial_randomness_authority=_KS_PROVENANCE["kappa_authority"],
            spatial_randomness_inputs=_KS_PROVENANCE["kappa_inputs"],
            warnings=warnings,
        )

    @classmethod
    def missing(
        cls,
        construct: ConstructLabel,
        reason: str,
    ) -> ConstructCertificate:
        """Create a certificate for a missing/unavailable construct.

        ADR-020 D8: missing kappa = NaN + warning, never 0.0.
        Provenance fields are explicitly None.
        """
        return cls(
            construct=construct,
            target_column=CONSTRUCT_TARGET_COLUMNS.get(construct, ""),
            forward_score=float("nan"),
            spatial_randomness=float("nan"),
            spatial_recoverability=float("nan"),
            morans_i=float("nan"),
            n_regions=0,
            n_observations=0,
            n_finite_targets=0,
            target_available=False,
            spatial_recoverability_source=None,
            spatial_recoverability_formula=None,
            spatial_recoverability_authority=None,
            spatial_recoverability_inputs=None,
            spatial_randomness_source=None,
            spatial_randomness_formula=None,
            spatial_randomness_authority=None,
            spatial_randomness_inputs=None,
            warnings=(reason,),
        )
