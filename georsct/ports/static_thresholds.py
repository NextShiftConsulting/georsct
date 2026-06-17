"""Static threshold source — ships with georsct, no external dependencies.

Preset values are snapshots from yrsn-controlplane (2026-06-17).
For canonical live values, use ControlPlaneThresholdSource or
APIThresholdSource adapters instead.
"""

from __future__ import annotations

from typing import Optional

from georsct.ports.threshold_source import ThresholdPreset, ThresholdSource

# ---------------------------------------------------------------------------
# Frozen preset snapshots
# ---------------------------------------------------------------------------

_GEOSPATIAL_CONUS27 = ThresholdPreset(
    preset_id="GEOSPATIAL_CONUS27",
    N_thr=0.50, alpha_min=0.30,
    c_min=0.30, gate_2_require_coherence=False,
    sigma_thr=0.50, kappa_base=0.22, lambda_turbulence=0.0,
    epsilon_L=0.01, oobleck_steepness=10.0, oobleck_sigma_c=0.35,
    landauer_sigma_tiebreaker=0.40,
    enable_gate_3b=False, r_bar_min=0.50,
    kappa_L_min=0.15,
)

_UNIVERSAL = ThresholdPreset(
    preset_id="UNIVERSAL",
    N_thr=0.50, alpha_min=0.30,
    c_min=0.40, gate_2_require_coherence=True,
    sigma_thr=0.50, kappa_base=0.50, lambda_turbulence=0.40,
    epsilon_L=0.05, oobleck_steepness=10.0, oobleck_sigma_c=0.35,
    landauer_sigma_tiebreaker=0.40,
    enable_gate_3b=True, r_bar_min=0.65,
    kappa_L_min=0.30,
)

_STRICT = ThresholdPreset(
    preset_id="STRICT",
    N_thr=0.35, alpha_min=0.50,
    c_min=0.50, gate_2_require_coherence=True,
    sigma_thr=0.35, kappa_base=0.60, lambda_turbulence=0.50,
    epsilon_L=0.03, oobleck_steepness=10.0, oobleck_sigma_c=0.35,
    landauer_sigma_tiebreaker=0.40,
    enable_gate_3b=True, r_bar_min=0.75,
    kappa_L_min=0.50,
)

_PRESETS: dict[str, ThresholdPreset] = {
    "GEOSPATIAL_CONUS27": _GEOSPATIAL_CONUS27,
    "UNIVERSAL": _UNIVERSAL,
    "STRICT": _STRICT,
}


class StaticThresholdSource(ThresholdSource):
    """Built-in threshold source with frozen preset snapshots.

    Works without any external packages installed. Suitable for
    public users, CI, and offline environments.
    """

    def get_preset(self, name: str) -> Optional[ThresholdPreset]:
        key = name.upper().replace("-", "_")
        return _PRESETS.get(key)

    def list_presets(self) -> list[str]:
        return list(_PRESETS.keys())
