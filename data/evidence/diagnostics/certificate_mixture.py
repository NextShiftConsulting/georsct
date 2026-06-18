#!/usr/bin/env python3
"""
certificate_mixture.py — Algebraic certificate blender for geo_cert.

Blends per-model proxy certificates into a single per-(zcta, task) allocation
certificate. Replaces the learned DyTopo router with an algebraic formula:

    weight_k = R_k * (1 - S_proxy_k) * calibration_k

where:
  R_k = model k's aggregate R² (representation strength)
  S_proxy_k = per-prediction superfluous component (prediction-level noise)
  calibration_k = model k's calibration score (1.0 in v1)

Design constraints:
  - Shared floor: every model gets at least `floor_weight` (prevents zero-out)
  - Dominance cap: no model exceeds `max_weight_share` of total (prevents
    single-model collapse)
  - Calibration slot: omega maps to calibration confidence (1.0 = fully
    calibrated, 0.5 = uninformative). Ready for v2 calibration integration.
  - No learned parameters: the blender is algebraic, not trained.

Output: one YRSNCertificate per (zcta, task) representing the blended view,
plus the per-model weights used and the blended R, S, N values.

Usage:
    from apps.geo_cert.certificates.mixture import CertificateMixer

    mixer = CertificateMixer(cert_rows)
    blended = mixer.blend_all()
    df = mixer.to_dataframe(blended)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from yrsn.core.certificates import YRSNCertificate

from apps.geo_cert.certificates.issuer import GeoCertRow, _make_cert, _clamp

log = logging.getLogger(__name__)

# Default blender parameters
FLOOR_WEIGHT = 0.05        # Minimum weight per model (prevents zero-out)
MAX_WEIGHT_SHARE = 0.60    # Maximum weight share (prevents single-model collapse)


@dataclass
class BlendedCertificate:
    """Blended certificate for a single (zcta, task) pair."""
    zcta: str
    task: str
    n_models: int
    # Blended simplex
    R_blended: float
    S_blended: float
    N_blended: float
    cert: YRSNCertificate
    # Per-model weights (normalized)
    model_weights: dict     # {model_version: weight}
    # Per-model raw scores before normalization
    raw_scores: dict        # {model_version: raw_weight}
    # Blended prediction
    y_pred_blended: float
    y_true: float


class CertificateMixer:
    """Blend per-model proxy certificates into allocation certificates.

    Args:
        cert_rows: list of GeoCertRow from GeoCertIssuer.
        floor_weight: minimum normalized weight per model.
        max_weight_share: maximum normalized weight per model.
    """

    def __init__(
        self,
        cert_rows: list,
        floor_weight: float = FLOOR_WEIGHT,
        max_weight_share: float = MAX_WEIGHT_SHARE,
    ):
        self.rows = cert_rows
        self.floor = floor_weight
        self.cap = max_weight_share

        # Index rows by (zcta, task)
        self._by_location = {}
        for r in cert_rows:
            key = (r.zcta, r.task)
            self._by_location.setdefault(key, []).append(r)

        n_locations = len(self._by_location)
        tasks = set(r.task for r in cert_rows)
        models = set(r.model_version for r in cert_rows)
        log.info(
            f"CertificateMixer: {n_locations} (zcta, task) pairs, "
            f"{len(tasks)} tasks, {len(models)} models"
        )

    def _compute_raw_weight(self, row: GeoCertRow) -> float:
        """Compute raw blending weight for a single model-prediction.

        weight_k = R_k * (1 - S_proxy_k) * calibration_k

        R_k is the model's aggregate R² (not per-prediction).
        S_proxy_k is the per-prediction superfluous component.
        calibration_k is the model's calibration score.
        """
        r_k = _clamp(row.r2_model)
        s_proxy_k = _clamp(row.S_proxy)
        cal_k = _clamp(row.calibration)
        return r_k * (1.0 - s_proxy_k) * cal_k

    def _normalize_weights(self, raw: dict) -> dict:
        """Normalize raw weights with floor and cap constraints.

        1. Apply floor: every model gets at least `floor_weight`
        2. Normalize to sum = 1
        3. Apply cap: no model exceeds `max_weight_share`
        4. Re-normalize after capping (redistribute excess)

        Args:
            raw: {model_version: raw_weight}

        Returns:
            {model_version: normalized_weight} summing to 1.0
        """
        n = len(raw)
        if n == 0:
            return {}
        if n == 1:
            return {k: 1.0 for k in raw}

        # Step 1: floor
        floored = {k: max(v, self.floor) for k, v in raw.items()}

        # Step 2: normalize
        total = sum(floored.values())
        if total < 1e-12:
            uniform = 1.0 / n
            return {k: uniform for k in raw}
        normed = {k: v / total for k, v in floored.items()}

        # Step 3-4: cap and redistribute (iterate until stable)
        for _ in range(10):
            capped = {}
            excess = 0.0
            n_uncapped = 0
            for k, v in normed.items():
                if v > self.cap:
                    capped[k] = self.cap
                    excess += v - self.cap
                else:
                    capped[k] = v
                    n_uncapped += 1

            if excess < 1e-12:
                return capped

            # Redistribute excess proportionally among uncapped models
            uncapped_total = sum(
                v for k, v in capped.items() if v < self.cap
            )
            if uncapped_total < 1e-12:
                # All models at cap — redistribute uniformly
                share = excess / n
                normed = {k: v + share for k, v in capped.items()}
            else:
                normed = {}
                for k, v in capped.items():
                    if v >= self.cap:
                        normed[k] = v
                    else:
                        normed[k] = v + excess * (v / uncapped_total)

        return normed

    def blend_for_location(
        self, zcta: str, task: str
    ) -> BlendedCertificate:
        """Blend certificates for a single (zcta, task) pair.

        Args:
            zcta: ZCTA identifier.
            task: Task name.

        Returns:
            BlendedCertificate with blended simplex and weights.
        """
        rows = self._by_location.get((zcta, task), [])
        if not rows:
            raise ValueError(f"No certificates for ({zcta}, {task})")

        # Compute raw weights
        raw_scores = {}
        for r in rows:
            raw_scores[r.model_version] = self._compute_raw_weight(r)

        # Normalize with floor + cap
        weights = self._normalize_weights(raw_scores)

        # Blend proxy simplex values (weighted average)
        R_blend = sum(weights[r.model_version] * r.R_proxy for r in rows)
        S_blend = sum(weights[r.model_version] * r.S_proxy for r in rows)
        N_blend = sum(weights[r.model_version] * r.N_proxy for r in rows)

        # Renormalize to ensure simplex
        total = R_blend + S_blend + N_blend
        if total > 0 and abs(total - 1.0) > 1e-9:
            R_blend /= total
            S_blend /= total
            N_blend /= total

        # Blended prediction (weighted average)
        y_pred_blend = sum(
            weights[r.model_version] * r.y_pred for r in rows
        )

        # Blended calibration (weighted average)
        cal_blend = sum(
            weights[r.model_version] * r.calibration for r in rows
        )

        cert = _make_cert(R_blend, S_blend, N_blend, omega=cal_blend)

        return BlendedCertificate(
            zcta=zcta,
            task=task,
            n_models=len(rows),
            R_blended=round(R_blend, 6),
            S_blended=round(S_blend, 6),
            N_blended=round(N_blend, 6),
            cert=cert,
            model_weights={k: round(v, 6) for k, v in weights.items()},
            raw_scores={k: round(v, 6) for k, v in raw_scores.items()},
            y_pred_blended=round(y_pred_blend, 6),
            y_true=rows[0].y_true,  # Same y_true for all models at this location
        )

    def blend_all(self) -> list:
        """Blend certificates for all (zcta, task) pairs.

        Returns:
            list of BlendedCertificate.
        """
        results = []
        for (zcta, task) in sorted(self._by_location.keys()):
            results.append(self.blend_for_location(zcta, task))

        log.info(f"Blended {len(results)} certificates")
        return results

    def to_dataframe(self, blended: list) -> pd.DataFrame:
        """Convert blended certificates to a flat DataFrame."""
        records = []
        for b in blended:
            rec = {
                "zcta": b.zcta,
                "task": b.task,
                "n_models": b.n_models,
                "R_blended": b.R_blended,
                "S_blended": b.S_blended,
                "N_blended": b.N_blended,
                "alpha": b.cert.alpha,
                "omega": b.cert.omega,
                "tau": b.cert.tau,
                "y_pred_blended": b.y_pred_blended,
                "y_true": b.y_true,
            }
            # Add per-model weights as columns
            for mv, w in b.model_weights.items():
                rec[f"w_{mv}"] = w
            records.append(rec)
        return pd.DataFrame(records)

    def summary(self, blended: list) -> dict:
        """Aggregate blending statistics per task."""
        df = self.to_dataframe(blended)
        summary = {}
        for task, group in df.groupby("task"):
            # Compute blend R² from blended predictions
            y_true = group["y_true"].values
            y_pred = group["y_pred_blended"].values
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            blend_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Weight concentration: max weight across models
            w_cols = [c for c in group.columns if c.startswith("w_")]
            max_weight = group[w_cols].max(axis=1).mean() if w_cols else 0.0

            summary[task] = {
                "n_certs": len(group),
                "R_blended_mean": round(float(group["R_blended"].mean()), 4),
                "S_blended_mean": round(float(group["S_blended"].mean()), 4),
                "N_blended_mean": round(float(group["N_blended"].mean()), 4),
                "alpha_mean": round(float(group["alpha"].mean()), 4),
                "tau_mean": round(float(group["tau"].mean()), 4),
                "blend_r2": round(blend_r2, 4),
                "mean_max_weight": round(float(max_weight), 4),
            }
        return summary

    @staticmethod
    def print_summary(summary: dict) -> None:
        """Print blending summary to stdout."""
        print("\n" + "=" * 90)
        print("CERTIFICATE BLENDING SUMMARY")
        print("=" * 90)

        hdr = (f"{'Task':<30} {'R_blend':>8} {'S_blend':>8} {'N_blend':>8} "
               f"{'alpha':>8} {'tau':>8} {'R2_blend':>8} {'max_w':>8}")
        print(hdr)
        print("-" * len(hdr))

        for task in sorted(summary.keys()):
            s = summary[task]
            print(
                f"{task:<30} {s['R_blended_mean']:>8.4f} "
                f"{s['S_blended_mean']:>8.4f} {s['N_blended_mean']:>8.4f} "
                f"{s['alpha_mean']:>8.4f} {s['tau_mean']:>8.4f} "
                f"{s['blend_r2']:>8.4f} {s['mean_max_weight']:>8.4f}"
            )

        print("=" * 90)
