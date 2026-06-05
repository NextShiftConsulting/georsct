#!/usr/bin/env python3
"""
certificate_issuer.py — Issue YRSNCertificates for geo_cert model predictions.

Maps geo_cert model performance onto the RSCT simplex:
  R = model R² (representation adequacy)
  S = 1 - R - TRF  (supportive structure, computed from the other two)
  TRF = task_residual_floor[task] or N_proxy[zcta, task, model]

Two certificate modes per (zcta, task, model):
  - ceiling: uses task_residual_floor[task] (invariant, for paper)
  - proxy:   uses N_proxy = 1 - R² per prediction (operational, for allocator)

Emits frozen YRSNCertificate instances from the yrsn core package.

Usage:
    from apps.geo_cert.certificates.issuer import GeoCertIssuer

    issuer = GeoCertIssuer(trf_table, oof_df)
    certs = issuer.issue_all()
    ceiling_certs = issuer.issue_ceiling()
    proxy_certs = issuer.issue_proxy()
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from yrsn.core.certificates import YRSNCertificate

log = logging.getLogger(__name__)


@dataclass
class GeoCertRow:
    """Single geo_cert certificate row with both decompositions."""
    zcta: str
    task: str
    model_version: str
    fold: str
    y_true: float
    y_pred: float
    r2_model: float
    # Ceiling decomposition (paper)
    R_ceiling: float
    S_ceiling: float
    task_residual_floor: float
    cert_ceiling: YRSNCertificate
    # Proxy decomposition (operational)
    R_proxy: float
    S_proxy: float
    N_proxy: float
    cert_proxy: YRSNCertificate
    # Calibration (1.0 in v1)
    calibration: float


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _make_cert(R: float, S: float, N: float, omega: float = 1.0) -> YRSNCertificate:
    """Build a YRSNCertificate from R, S, N.

    alpha = R / (R + N), clamped to avoid division by zero.
    tau = 1 / alpha_omega, where alpha_omega = omega * alpha + (1-omega) * prior.
    omega maps to calibration confidence (1.0 = fully calibrated in v1).
    """
    R = _clamp(R)
    S = _clamp(S)
    N = _clamp(N)

    # Enforce simplex: renormalize if sum != 1 due to floating point
    total = R + S + N
    if total > 0 and abs(total - 1.0) > 1e-9:
        R, S, N = R / total, S / total, N / total

    alpha = R / (R + N) if (R + N) > 1e-12 else 0.5
    prior = 0.5
    alpha_omega = omega * alpha + (1.0 - omega) * prior
    tau = 1.0 / max(alpha_omega, 1e-12)

    return YRSNCertificate(
        R=round(R, 6),
        S=round(S, 6),
        N=round(N, 6),
        alpha=round(alpha, 6),
        omega=round(omega, 6),
        tau=round(tau, 6),
    )


EPS = 1e-12


def compute_simplex_diagnostic_ratios(
    R: float,
    S_sup: float,
    N: float,
) -> dict:
    """Compute diagnostic simplex ratio views (ADR-021).

    These are diagnostic only.
    They do not alter canonical alpha, kappa_compat, or runtime gate decisions.
    Log under diagnostics.simplex_ratios, not certificate.core.
    """
    return {
        "alpha": R / max(R + N, EPS),
        "SCR_norm": R / max(R + S_sup, EPS),
        "CNR_norm": S_sup / max(S_sup + N, EPS),
        "leaderboard_scalar": R + S_sup,
    }


class GeoCertIssuer:
    """Issue YRSNCertificates for geo_cert OOF predictions.

    Args:
        trf_table: {task: task_residual_floor} from the TRF estimator.
        oof_df: OOF predictions DataFrame in ceiling_schema format.
        model_r2_table: Optional {(task, model_version): R²} for aggregate R.
            If None, R² is computed per-model from oof_df.
        calibration: Per-model calibration scores. Default 1.0 for all (v1).
    """

    def __init__(
        self,
        trf_table: dict,
        oof_df: pd.DataFrame,
        model_r2_table: dict = None,
        calibration: dict = None,
    ):
        self.trf_table = trf_table
        self.oof = oof_df
        self.calibration = calibration or {}

        # Compute per-(task, model) R² from OOF if not provided
        if model_r2_table is not None:
            self.model_r2 = model_r2_table
        else:
            self.model_r2 = self._compute_model_r2()

        self.tasks = sorted(oof_df["task"].unique())
        self.models = sorted(oof_df["model_version"].unique())
        log.info(f"GeoCertIssuer: {len(self.tasks)} tasks, "
                 f"{len(self.models)} models, "
                 f"{len(trf_table)} TRF entries")

    def _compute_model_r2(self) -> dict:
        """Compute per-(task, model_version) R² from OOF predictions."""
        r2_table = {}
        for (task, mv), group in self.oof.groupby(["task", "model_version"]):
            y_true = group["y_true"].values
            y_pred = group["y_pred"].values
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_table[(task, mv)] = _clamp(r2)
        return r2_table

    def _get_calibration(self, model_version: str) -> float:
        """Get calibration score for a model. Default 1.0 (v1)."""
        return self.calibration.get(model_version, 1.0)

    def issue_for_row(
        self,
        zcta: str,
        task: str,
        model_version: str,
        fold: str,
        y_true: float,
        y_pred: float,
    ) -> GeoCertRow:
        """Issue a certificate pair (ceiling + proxy) for one prediction."""
        # Ceiling decomposition: R = model R², TRF = task_residual_floor[task]
        r2_model = self.model_r2.get((task, model_version), 0.0)
        trf_val = self.trf_table.get(task)
        if trf_val is None:
            raise ValueError(
                f"No task_residual_floor for task '{task}'. "
                f"Available: {sorted(self.trf_table.keys())}"
            )

        R_ceiling = _clamp(r2_model)
        trf = _clamp(trf_val)
        S_ceiling = _clamp(1.0 - R_ceiling - trf)

        calibration = self._get_calibration(model_version)
        cert_ceiling = _make_cert(R_ceiling, S_ceiling, trf, omega=calibration)

        # Proxy decomposition: per-prediction residual
        residual_sq = (y_true - y_pred) ** 2
        var_y = np.var(self.oof[self.oof["task"] == task]["y_true"].values)
        if var_y > 1e-12:
            # Per-prediction proxy: fraction of variance explained
            R_proxy = _clamp(1.0 - residual_sq / var_y)
        else:
            R_proxy = 0.0
        N_proxy = _clamp(1.0 - R_proxy)
        S_proxy = _clamp(1.0 - R_proxy - N_proxy)  # 0 by construction

        cert_proxy = _make_cert(R_proxy, S_proxy, N_proxy, omega=calibration)

        return GeoCertRow(
            zcta=zcta,
            task=task,
            model_version=model_version,
            fold=fold,
            y_true=y_true,
            y_pred=y_pred,
            r2_model=r2_model,
            R_ceiling=R_ceiling,
            S_ceiling=S_ceiling,
            task_residual_floor=trf,
            cert_ceiling=cert_ceiling,
            R_proxy=R_proxy,
            S_proxy=S_proxy,
            N_proxy=N_proxy,
            cert_proxy=cert_proxy,
            calibration=calibration,
        )

    def issue_all(self) -> list:
        """Issue certificates for every row in the OOF DataFrame.

        Returns list of GeoCertRow.
        """
        rows = []
        for _, r in self.oof.iterrows():
            row = self.issue_for_row(
                zcta=r["zcta"],
                task=r["task"],
                model_version=r["model_version"],
                fold=r["fold"],
                y_true=r["y_true"],
                y_pred=r["y_pred"],
            )
            rows.append(row)

        log.info(f"Issued {len(rows)} certificate pairs "
                 f"({len(rows)} ceiling + {len(rows)} proxy)")
        return rows

    def to_dataframe(self, rows: list) -> pd.DataFrame:
        """Convert GeoCertRow list to a flat DataFrame for analysis."""
        records = []
        for r in rows:
            records.append({
                "zcta": r.zcta,
                "task": r.task,
                "model_version": r.model_version,
                "fold": r.fold,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "r2_model": r.r2_model,
                "R_ceiling": r.R_ceiling,
                "S_ceiling": r.S_ceiling,
                "task_residual_floor": r.task_residual_floor,
                "R_proxy": r.R_proxy,
                "S_proxy": r.S_proxy,
                "N_proxy": r.N_proxy,
                "calibration": r.calibration,
                "alpha_ceiling": r.cert_ceiling.alpha,
                "omega": r.cert_ceiling.omega,
                "tau_ceiling": r.cert_ceiling.tau,
                "alpha_proxy": r.cert_proxy.alpha,
                "tau_proxy": r.cert_proxy.tau,
            })
        return pd.DataFrame(records)

    def summary(self, rows: list) -> dict:
        """Aggregate certificate statistics per task."""
        df = self.to_dataframe(rows)
        summary = {}
        for task, group in df.groupby("task"):
            summary[task] = {
                "n_certs": len(group),
                "n_models": group["model_version"].nunique(),
                "R_ceiling_mean": round(float(group["R_ceiling"].mean()), 4),
                "S_ceiling_mean": round(float(group["S_ceiling"].mean()), 4),
                "task_residual_floor": round(float(group["task_residual_floor"].iloc[0]), 4),
                "R_proxy_mean": round(float(group["R_proxy"].mean()), 4),
                "alpha_ceiling_mean": round(float(group["alpha_ceiling"].mean()), 4),
            }
        return summary
