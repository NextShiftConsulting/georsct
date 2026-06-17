"""Threshold source port — ABC for resolving gate threshold presets.

Gate thresholds are consumed by healthcheck and evaluation layers.
Concrete implementations belong in adapters (e.g., controlplane adapter,
API adapter) or ship as built-in static defaults.

Import rule: this module imports nothing outside georsct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThresholdPreset:
    """Gate thresholds + diagnostic thresholds as a frozen value object.

    Gate-related fields correspond to yrsn-controlplane GatekeeperConfig.
    Diagnostic and trajectory fields are healthcheck-local additions.
    """

    preset_id: str

    # --- Gate thresholds ---
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
        import math

        x = self.oobleck_steepness * (sigma - self.oobleck_sigma_c)
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            s = math.exp(x) / (1.0 + math.exp(x))
        return self.kappa_base + self.lambda_turbulence * s


class ThresholdSource(ABC):
    """Port: resolves named threshold presets.

    Implementations:
      - StaticThresholdSource (ships with georsct — hardcoded defaults)
      - ControlPlaneThresholdSource (adapter — reads yrsn_controlplane)
      - APIThresholdSource (adapter — reads api.swarms.network)
    """

    @abstractmethod
    def get_preset(self, name: str) -> Optional[ThresholdPreset]:
        """Resolve a preset by name. Returns None if unknown."""

    @abstractmethod
    def list_presets(self) -> list[str]:
        """List available preset names."""
