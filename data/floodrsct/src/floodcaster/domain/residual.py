"""Residual analysis domain functions.

Pure functions -- no I/O.
Spatial residual diagnostics feed back into turbulence scoring.
"""

import numpy as np
import pandas as pd


def compute_residual_spatial_lag(
    residuals: np.ndarray,
    w: "libpysal.weights.W",
) -> np.ndarray:
    """Spatially lagged residuals for autocorrelation diagnostics.

    Pattern: AGDS Ch.9 (residual Moran scatter plot).
    """
    from pysal.lib.weights.spatial_lag import lag_spatial

    return lag_spatial(w, residuals)


def flag_residual_outliers(
    residuals: pd.Series,
    threshold_sigma: float = 2.0,
) -> pd.Series:
    """Boolean mask of residual outliers beyond threshold.

    Returns True for observations where |residual| > threshold * std.
    """
    std = residuals.std()
    if std < 1e-12:
        return pd.Series(False, index=residuals.index)
    return residuals.abs() > (threshold_sigma * std)
