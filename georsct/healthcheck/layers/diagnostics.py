"""Layer 3: Diagnostic ratio analysis."""

from __future__ import annotations

from typing import Any

from ..models import DiagnosticResult
from ..thresholds import ThresholdPreset


def analyze_diagnostics(
    diag: dict[str, Any] | None,
    gearbox: dict[str, Any] | None,
    preset: ThresholdPreset,
) -> DiagnosticResult | None:
    """Analyze diagnostic ratios and gearbox warmup signals.

    Args:
        diag: Diagnostics dict for this cell/level (may be None).
        gearbox: Gearbox warmup dict for this cell (may be None).
        preset: Threshold preset for warning thresholds.

    Returns:
        DiagnosticResult, or None if no diagnostic data available.
    """
    if diag is None and gearbox is None:
        return None

    warnings: list[str] = []
    leakage = None
    transfer = None
    solver = None
    residual_spatial = None

    if diag is not None:
        leakage = diag.get("diag_leakage")
        transfer = diag.get("diag_transfer")
        solver = diag.get("diag_solver")
        residual_spatial = diag.get("diag_residual_spatial")

        if leakage is not None and leakage > preset.leakage_warn:
            warnings.append(
                f"SPATIAL_BLEED: leakage={leakage:.3f} > {preset.leakage_warn}"
            )

        if transfer is None or transfer == 0.0:
            warnings.append("NO_TRANSFER_SIGNAL: no cross-scenario signal detected")

        if solver is not None and solver < preset.solver_warn:
            warnings.append(
                f"WEAK_SOLVER: solver={solver:.3f} < {preset.solver_warn}"
            )

        if residual_spatial is not None and residual_spatial > preset.residual_spatial_warn:
            warnings.append(
                f"UNEXPLAINED_SPATIAL_PATTERN: residual_spatial={residual_spatial:.3f} "
                f"> {preset.residual_spatial_warn}"
            )

    if gearbox is not None:
        collapse_risk = gearbox.get("collapse_risk")
        coherence = gearbox.get("coherence")

        if collapse_risk is not None and collapse_risk >= preset.collapse_risk_warn:
            warnings.append(
                f"COLLAPSE_RISK: collapse_risk={collapse_risk:.2f} >= {preset.collapse_risk_warn}"
            )

        if coherence is not None and coherence < preset.coherence_warn:
            warnings.append(
                f"LOW_COHERENCE: coherence={coherence:.3f} < {preset.coherence_warn}"
            )

    return DiagnosticResult(
        leakage=leakage,
        transfer=transfer,
        solver=solver,
        residual_spatial=residual_spatial,
        warnings=warnings,
    )
