"""Threshold presets sourced from yrsn-controlplane (single source of truth).

Gate thresholds are imported from yrsn_controlplane.GatekeeperConfig.
Diagnostic and trajectory thresholds are healthcheck-local additions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from yrsn_controlplane import GatekeeperConfig, get_preset


@dataclass(frozen=True)
class ThresholdPreset:
    """Gate thresholds from yrsn-controlplane + diagnostic-only extras.

    Gate-related fields are sourced from a canonical GatekeeperConfig
    to prevent threshold drift between healthcheck and production.
    """

    preset_id: str

    # --- Gate thresholds (from GatekeeperConfig) ---
    # Gate 1
    N_thr: float = 0.50
    alpha_min: float = 0.30

    # Gate 2
    c_min: float = 0.40
    gate_2_require_coherence: bool = True

    # Gate 3 -- Oobleck sigmoidal
    sigma_thr: float = 0.50
    kappa_base: float = 0.50
    lambda_turbulence: float = 0.40
    epsilon_L: float = 0.05
    oobleck_steepness: float = 10.0
    oobleck_sigma_c: float = 0.35
    landauer_sigma_tiebreaker: float = 0.40

    # Gate 3B
    enable_gate_3b: bool = True
    r_bar_min: float = 0.65

    # Gate 4
    kappa_L_min: float = 0.30

    # --- Diagnostic thresholds (healthcheck-only) ---
    leakage_warn: float = 1.05
    solver_warn: float = 0.50
    residual_spatial_warn: float = 0.40
    collapse_risk_warn: float = 0.40
    coherence_warn: float = 0.30

    # --- Trajectory thresholds (healthcheck-only) ---
    trend_delta: float = 0.02
    convergence_uplift_pct: float = 1.0
    converged_alpha_min: float = 0.50
    converged_kappa_min: float = 0.50

    def kappa_req(self, sigma: float) -> float:
        """Oobleck sigmoidal kappa requirement for a given sigma.

        kappa_req(sigma) = kappa_base + lambda_turbulence * sigmoid(steepness * (sigma - sigma_c))
        """
        x = self.oobleck_steepness * (sigma - self.oobleck_sigma_c)
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            s = math.exp(x) / (1.0 + math.exp(x))
        return self.kappa_base + self.lambda_turbulence * s


# ---------------------------------------------------------------------------
# Factory: build ThresholdPreset from canonical GatekeeperConfig
# ---------------------------------------------------------------------------

def _from_config(name: str, cfg: GatekeeperConfig) -> ThresholdPreset:
    """Build a ThresholdPreset from a canonical GatekeeperConfig."""
    return ThresholdPreset(
        preset_id=name,
        # Gate 1
        N_thr=cfg.N_thr,
        alpha_min=cfg.alpha_min,
        # Gate 2
        c_min=cfg.c_min,
        gate_2_require_coherence=cfg.gate_2_require_coherence,
        # Gate 3
        sigma_thr=cfg.sigma_thr,
        kappa_base=cfg.kappa_base,
        lambda_turbulence=cfg.lambda_turbulence,
        epsilon_L=cfg.epsilon_L,
        oobleck_steepness=cfg.steepness,
        oobleck_sigma_c=cfg.sigma_c,
        landauer_sigma_tiebreaker=cfg.landauer_sigma_tiebreaker,
        # Gate 3B
        enable_gate_3b=cfg.enable_gate_3b,
        r_bar_min=cfg.r_bar_min,
        # Gate 4
        kappa_L_min=cfg.kappa_L_min,
    )


# ---------------------------------------------------------------------------
# Named presets (sourced from yrsn-controlplane)
# ---------------------------------------------------------------------------

GEOSPATIAL_CONUS27 = _from_config("GEOSPATIAL_CONUS27", get_preset("geospatial-conus27"))
UNIVERSAL = _from_config("UNIVERSAL", get_preset("universal"))
STRICT = _from_config("STRICT", get_preset("strict"))

PRESETS: dict[str, ThresholdPreset] = {
    "GEOSPATIAL_CONUS27": GEOSPATIAL_CONUS27,
    "UNIVERSAL": UNIVERSAL,
    "STRICT": STRICT,
}


def resolve_preset(name: str | None) -> ThresholdPreset:
    """Resolve a preset by name. Falls back to UNIVERSAL."""
    if name is None:
        return UNIVERSAL
    key = name.upper().replace("-", "_")
    return PRESETS.get(key, UNIVERSAL)
