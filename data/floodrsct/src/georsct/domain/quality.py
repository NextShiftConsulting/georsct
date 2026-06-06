"""Quality scoring via spatial regression diagnostics.

Pure functions -- no I/O, no S3, no SQL.
Source patterns: AGDS Ch.9 (OLS, GWR, MGWR, spatial fixed effects).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Summary diagnostics from spatial regression."""

    r2: float
    adj_r2: float
    residual_moran_i: Optional[float]    # spatial autocorrelation in residuals
    residual_moran_p: Optional[float]
    aic: Optional[float]
    n_obs: int


@dataclass(frozen=True)
class GWRResult:
    """Geographically weighted regression output."""

    bandwidth: float
    local_r2: np.ndarray          # per-observation R^2
    local_coefficients: np.ndarray  # (n_obs, n_vars) coefficient surface
    residuals: np.ndarray
    aic: float


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    var_names: list[str],
) -> RegressionDiagnostics:
    """Baseline OLS regression.

    Pattern: spreg.OLS (AGDS Ch.9).
    """
    from pysal.model import spreg

    ols = spreg.OLS(
        y.reshape(-1, 1),
        X,
        name_y="target",
        name_x=var_names,
    )
    return RegressionDiagnostics(
        r2=ols.r2,
        adj_r2=ols.ar2,
        residual_moran_i=None,
        residual_moran_p=None,
        aic=None,
        n_obs=len(y),
    )


def fit_ols_regimes(
    y: np.ndarray,
    X: np.ndarray,
    regime_var: list[str],
    var_names: list[str],
) -> tuple[RegressionDiagnostics, float]:
    """Spatial fixed effects via OLS Regimes.

    Pattern: spreg.OLS_Regimes (AGDS Ch.9).
    Returns (diagnostics, chow_joint_statistic).
    """
    from pysal.model import spreg

    sfe = spreg.OLS_Regimes(
        y.reshape(-1, 1),
        X,
        regime_var,
        constant_regi="many",
        cols2regi=[False] * X.shape[1],
        regime_err_sep=False,
        name_y="target",
        name_x=var_names,
    )
    diag = RegressionDiagnostics(
        r2=sfe.r2,
        adj_r2=sfe.ar2,
        residual_moran_i=None,
        residual_moran_p=None,
        aic=None,
        n_obs=len(y),
    )
    return diag, float(sfe.chow.joint)


def fit_gwr(
    coords: list[tuple[float, float]],
    y: np.ndarray,
    X: np.ndarray,
    bw_min: int = 2,
) -> GWRResult:
    """Geographically Weighted Regression with auto bandwidth selection.

    Pattern: mgwr.gwr.GWR + mgwr.sel_bw.Sel_BW (AGDS Ch.9).
    """
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    y_col = y.reshape(-1, 1)
    selector = Sel_BW(coords, y_col, X, spherical=True)
    bw = selector.search(bw_min=bw_min)

    result = GWR(coords, y_col, X, bw).fit()
    return GWRResult(
        bandwidth=float(bw),
        local_r2=result.localR2,
        local_coefficients=result.params,
        residuals=result.resid_response.flatten(),
        aic=float(result.aicc),
    )


def fit_mgwr(
    coords: list[tuple[float, float]],
    y: np.ndarray,
    X: np.ndarray,
    bw_min: int = 4,
) -> GWRResult:
    """Multi-scale GWR -- per-variable bandwidths.

    Pattern: mgwr.gwr.MGWR + multi=True Sel_BW (AGDS Ch.9).
    Inputs should be standardized (zero mean, unit variance).
    """
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW

    y_col = y.reshape(-1, 1)
    selector = Sel_BW(coords, y_col, X, multi=True, spherical=True)
    selector.search(multi_bw_min=[bw_min])

    result = MGWR(coords, y_col, X, selector, sigma2_v1=True).fit()
    return GWRResult(
        bandwidth=float(np.mean(selector.bw_)),
        local_r2=result.localR2,
        local_coefficients=result.params,
        residuals=result.resid_response.flatten(),
        aic=float(result.aicc),
    )
