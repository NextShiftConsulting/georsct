"""Turbulence scoring via spatial autocorrelation.

Pure functions -- no I/O, no S3, no SQL.
Source patterns: AGDS Ch.6 (Spatial Autocorrelation), esda Moran/Geary/LISA.
Existing impl:  compute_residual_lisa.py::compute_lisa() (SageMaker job).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurbulenceResult:
    """Global + local turbulence summary for a spatial domain."""

    moran_i: float
    moran_p: float
    geary_c: float
    geary_p: float
    n_hotspots: int       # HH clusters at alpha
    n_coldspots: int      # LL clusters at alpha
    n_outliers: int       # HL + LH at alpha
    fraction_significant: float


QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def compute_global_autocorrelation(
    values: np.ndarray,
    w: "libpysal.weights.W",
    permutations: int = 999,
) -> tuple[float, float, float, float]:
    """Return (moran_I, moran_p, geary_C, geary_p).

    Pattern: esda.moran.Moran + esda.geary.Geary (AGDS Ch.6).
    """
    from pysal.explore.esda.moran import Moran
    from pysal.explore.esda.geary import Geary

    m = Moran(values, w, permutations=permutations)
    g = Geary(values, w, permutations=permutations)
    return m.I, m.p_sim, g.C, g.p_sim


def compute_lisa_clusters(
    values: np.ndarray,
    w: "libpysal.weights.W",
    permutations: int = 999,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Local Moran's I with quadrant classification.

    Returns DataFrame with columns: local_i, p_value, quadrant, significant.
    Pattern: esda.moran.Moran_Local (AGDS Ch.6).
    Existing: compute_residual_lisa.py::compute_lisa().
    """
    from pysal.explore.esda.moran import Moran_Local

    lisa = Moran_Local(values, w, permutations=permutations)

    df = pd.DataFrame({
        "local_i": lisa.Is,
        "p_value": lisa.p_sim,
        "quadrant": [QUADRANT_LABELS.get(q, "NS") for q in lisa.q],
        "significant": lisa.p_sim < alpha,
    })
    return df


def score_turbulence(
    values: np.ndarray,
    w: "libpysal.weights.W",
    permutations: int = 999,
    alpha: float = 0.05,
) -> TurbulenceResult:
    """Full turbulence assessment: global stats + local cluster counts.

    Combines Moran's I, Geary's C, and LISA into a single result object.
    """
    mi, mp, gc, gp = compute_global_autocorrelation(values, w, permutations)
    clusters = compute_lisa_clusters(values, w, permutations, alpha)

    sig = clusters[clusters["significant"]]
    return TurbulenceResult(
        moran_i=mi,
        moran_p=mp,
        geary_c=gc,
        geary_p=gp,
        n_hotspots=int((sig["quadrant"] == "HH").sum()),
        n_coldspots=int((sig["quadrant"] == "LL").sum()),
        n_outliers=int(sig["quadrant"].isin(["HL", "LH"]).sum()),
        fraction_significant=float(clusters["significant"].mean()),
    )
