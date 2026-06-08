"""Layer 4: Cross-level trajectory analysis."""

from __future__ import annotations

from typing import Any

from ..models import TrajectoryResult
from ..thresholds import ThresholdPreset


def _trend(prev: float, curr: float, delta_thr: float) -> str:
    d = curr - prev
    if d > delta_thr:
        return "improving"
    if d < -delta_thr:
        return "regressing"
    return "stalled"


def _sigma_trend(prev: float, curr: float, delta_thr: float) -> str:
    d = curr - prev
    if d < -delta_thr:
        return "stabilizing"
    if d > delta_thr:
        return "destabilizing"
    return "flat"


def analyze_trajectory(
    certs_by_level: dict[str, dict[str, Any]],
    money_row: dict[str, Any] | None,
    provenance: dict[str, str],
    preset: ThresholdPreset,
) -> TrajectoryResult | None:
    """Analyze cross-level trajectory for a single cell.

    Args:
        certs_by_level: Certificate dicts keyed by level (e.g., {"r0": {...}, "r1": {...}}).
        money_row: Money table row for this cell (may be None).
        provenance: Preset IDs keyed by level.
        preset: Threshold preset for trend/convergence thresholds.

    Returns:
        TrajectoryResult, or None if fewer than 2 levels.
    """
    levels = sorted(certs_by_level.keys())
    if len(levels) < 2:
        return None

    warnings: list[str] = []

    # Provenance check
    preset_ids = {lvl: provenance.get(lvl, "unknown") for lvl in levels}
    unique_presets = set(preset_ids.values()) - {"unknown"}
    provenance_valid = len(unique_presets) <= 1

    if not provenance_valid:
        warnings.append(
            f"PROVENANCE_MISMATCH: levels span presets {preset_ids}"
        )
        return TrajectoryResult(
            levels_present=levels,
            alpha_trend="unknown",
            sigma_trend="unknown",
            uplift_pct={},
            convergence_state="unknown",
            warnings=warnings,
            provenance_valid=False,
        )

    # Compute trends on last step only
    prev_level = levels[-2]
    curr_level = levels[-1]
    prev_cert = certs_by_level[prev_level]
    curr_cert = certs_by_level[curr_level]

    prev_alpha = prev_cert.get("alpha", 0.0)
    curr_alpha = curr_cert.get("alpha", 0.0)
    prev_sigma = prev_cert.get("sigma", 0.0)
    curr_sigma = curr_cert.get("sigma", 0.0)
    curr_kappa = curr_cert.get("kappa", curr_cert.get("kappa_compat", 0.0))

    alpha_t = _trend(prev_alpha, curr_alpha, preset.trend_delta)
    sigma_t = _sigma_trend(prev_sigma, curr_sigma, preset.trend_delta)

    if alpha_t == "regressing":
        warnings.append(
            f"TRAJECTORY_REGRESSION: alpha {prev_alpha:.3f} -> {curr_alpha:.3f}"
        )
    if sigma_t == "destabilizing":
        warnings.append(
            f"SIGMA_DESTABILIZING: sigma {prev_sigma:.3f} -> {curr_sigma:.3f}"
        )

    # Compute uplift
    uplift: dict[str, float] = {}
    for i in range(1, len(levels)):
        prev_l = levels[i - 1]
        curr_l = levels[i]
        key = f"{prev_l}->{curr_l}"

        # Try money table first
        mt_key = f"uplift_{prev_l}_{curr_l}_pct"
        if money_row and mt_key in money_row and money_row[mt_key] is not None:
            uplift[key] = money_row[mt_key]
        else:
            # Fall back to raw metric delta
            prev_metric = certs_by_level[prev_l].get(
                "spatial_metric", certs_by_level[prev_l].get("R", 0.0)
            )
            curr_metric = certs_by_level[curr_l].get(
                "spatial_metric", certs_by_level[curr_l].get("R", 0.0)
            )
            if (prev_metric is not None and curr_metric is not None
                    and prev_metric > 0):
                uplift[key] = round(
                    100.0 * (curr_metric - prev_metric) / prev_metric, 2
                )
            else:
                uplift[key] = 0.0

    # Convergence on last step
    last_key = f"{prev_level}->{curr_level}"
    last_uplift = uplift.get(last_key, 0.0)

    if last_uplift < 0:
        convergence_state = "regressing"
    elif abs(last_uplift) < preset.convergence_uplift_pct:
        if (curr_alpha >= preset.converged_alpha_min
                and curr_kappa >= preset.converged_kappa_min):
            convergence_state = "healthy_converged"
        else:
            convergence_state = "prematurely_stalled"
            warnings.append(
                f"PREMATURE_STALL: uplift={last_uplift:.1f}%, "
                f"alpha={curr_alpha:.3f}, kappa={curr_kappa:.3f}"
            )
    else:
        convergence_state = "active"

    return TrajectoryResult(
        levels_present=levels,
        alpha_trend=alpha_t,
        sigma_trend=sigma_t,
        uplift_pct=uplift,
        convergence_state=convergence_state,
        warnings=warnings,
        provenance_valid=provenance_valid,
    )
