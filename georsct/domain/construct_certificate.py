"""Per-construct certification for five-construct divergence.

Pure domain objects and computation -- no I/O, no S3, no pandas.

Each of five flood constructs (JRC, Deltares, FEMA, NFIP, FAST) encodes
a different aspect of flood risk.  This module defines the typed construct
labels, the per-construct certificate, and a pure Moran's I implementation
for kappa_spatial.

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
# kappa_spatial -- pure Moran's I (no pysal dependency)
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


def compute_kappa_spatial(
    residuals_by_region: np.ndarray,
    W_geo: sparse.csr_matrix,
) -> tuple[float, float]:
    """Compute kappa_spatial = 1 - |Moran's I| from region-level residuals.

    ADR-020 D8: returns (NaN, NaN) when data insufficient, never (0.0, 0.0).

    Args:
        residuals_by_region: (n_regions,) mean residual per region.
        W_geo: (n_regions, n_regions) row-normalized adjacency.

    Returns:
        (kappa_spatial, morans_i_raw)
    """
    I = _morans_i(residuals_by_region, W_geo)
    if not np.isfinite(I):
        return float("nan"), float("nan")
    kappa = float(np.clip(1.0 - abs(I), 0.0, 1.0))
    return kappa, float(I)


# ---------------------------------------------------------------------------
# Per-construct certificate (ADR-020 D8 provenance)
# ---------------------------------------------------------------------------

# Default provenance for kappa_reconstruct.
_KR_PROVENANCE = {
    "kappa_source": "georsct.domain.kappa_reconstruct",
    "kappa_formula": "1 - excess_crossings / max_possible",
    "kappa_authority": "RSCT-P008",
    "kappa_inputs": ("embeddings", "coords2d"),
}

# Default provenance for kappa_spatial.
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
    kappa_spatial: float
    kappa_reconstruct: float
    morans_i: float

    # Counts
    n_regions: int
    n_observations: int
    n_finite_targets: int

    # Availability
    target_available: bool

    # ADR-020 D8: kappa provenance (None on missing certificates)
    kappa_reconstruct_source: Optional[str] = None
    kappa_reconstruct_formula: Optional[str] = None
    kappa_reconstruct_authority: Optional[str] = None
    kappa_reconstruct_inputs: Optional[tuple[str, ...]] = None

    kappa_spatial_source: Optional[str] = None
    kappa_spatial_formula: Optional[str] = None
    kappa_spatial_authority: Optional[str] = None
    kappa_spatial_inputs: Optional[tuple[str, ...]] = None

    warnings: tuple[str, ...] = ()

    @classmethod
    def from_scores(
        cls,
        construct: ConstructLabel,
        target_column: str,
        forward_score: float,
        kappa_spatial: float,
        kappa_reconstruct: float,
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
            kappa_spatial=kappa_spatial,
            kappa_reconstruct=kappa_reconstruct,
            morans_i=morans_i,
            n_regions=n_regions,
            n_observations=n_observations,
            n_finite_targets=n_finite_targets,
            target_available=True,
            kappa_reconstruct_source=_KR_PROVENANCE["kappa_source"],
            kappa_reconstruct_formula=_KR_PROVENANCE["kappa_formula"],
            kappa_reconstruct_authority=_KR_PROVENANCE["kappa_authority"],
            kappa_reconstruct_inputs=_KR_PROVENANCE["kappa_inputs"],
            kappa_spatial_source=_KS_PROVENANCE["kappa_source"],
            kappa_spatial_formula=_KS_PROVENANCE["kappa_formula"],
            kappa_spatial_authority=_KS_PROVENANCE["kappa_authority"],
            kappa_spatial_inputs=_KS_PROVENANCE["kappa_inputs"],
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
            kappa_spatial=float("nan"),
            kappa_reconstruct=float("nan"),
            morans_i=float("nan"),
            n_regions=0,
            n_observations=0,
            n_finite_targets=0,
            target_available=False,
            kappa_reconstruct_source=None,
            kappa_reconstruct_formula=None,
            kappa_reconstruct_authority=None,
            kappa_reconstruct_inputs=None,
            kappa_spatial_source=None,
            kappa_spatial_formula=None,
            kappa_spatial_authority=None,
            kappa_spatial_inputs=None,
            warnings=(reason,),
        )
