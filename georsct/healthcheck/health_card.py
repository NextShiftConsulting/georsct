"""Compose layers into per-cell HealthCards and classify overall health."""

from __future__ import annotations

from typing import Any

from .models import (
    CellKey,
    DiagnosticResult,
    GateResult,
    HealthCard,
    RoutingResult,
    TrajectoryResult,
    WorkflowSnapshot,
)
from .layers.gate_triage import evaluate_gates
from .layers.degradation import classify_degradation
from .layers.diagnostics import analyze_diagnostics
from .layers.trajectory import analyze_trajectory
from .layers.routing import analyze_routing
from .loader import (
    get_admission_for_cell,
    get_cells,
    get_cert_for_cell,
    get_diag_for_cell,
    get_gearbox_for_cell,
    get_money_for_cell,
    get_routing_for_cell,
)
from .thresholds import ThresholdPreset


def _classify_health(
    gate: GateResult,
    diagnostics: DiagnosticResult | None,
    trajectory: TrajectoryResult | None,
    routing: RoutingResult | None,
) -> str:
    """Classify overall health. Priority: critical > failing > warning > healthy."""
    has_diag_warnings = diagnostics is not None and len(diagnostics.warnings) > 0
    has_traj_regression = (
        trajectory is not None and trajectory.alpha_trend == "regressing"
    )
    has_routing_conflict = (
        routing is not None and len(routing.conflicts) > 0
    )
    has_premature_stall = (
        trajectory is not None
        and trajectory.convergence_state == "prematurely_stalled"
    )
    gate_failed = gate.decision != "EXECUTE"

    # Critical
    if gate.decision in ("REJECT", "BLOCK"):
        return "critical"
    if gate_failed and has_traj_regression:
        return "critical"
    if gate_failed and has_diag_warnings:
        return "critical"

    # Failing
    if gate_failed:
        return "failing"

    # Warning
    if has_diag_warnings:
        return "warning"
    if has_traj_regression:
        return "warning"
    if has_premature_stall:
        return "warning"
    if has_routing_conflict:
        return "warning"

    return "healthy"


def _generate_next_steps(
    gate: GateResult,
    diagnostics: DiagnosticResult | None,
    trajectory: TrajectoryResult | None,
    routing: RoutingResult | None,
) -> list[str]:
    """Generate ordered recommendations from layer outputs."""
    steps: list[str] = []

    # Routing conflicts first (most surprising)
    if routing and routing.conflicts:
        for conflict in routing.conflicts:
            steps.append(conflict)

    # Gate failures
    if gate.decision == "REJECT":
        if gate.sub_signal == "N_FLOOR_BREACH":
            steps.append(
                "Noise saturation with low alpha -- both noise floor "
                "and discriminative quality failed; verify target has "
                "variance and check feature relevance"
            )
    elif gate.decision == "BLOCK":
        steps.append(
            "Low coherence -- folds disagree; "
            "check spatial split validity and sample size"
        )
    elif gate.decision == "RE_ENCODE":
        steps.append(
            f"Gate 3 RE_ENCODE -- kappa below Oobleck threshold at "
            f"sigma={gate.gate_evidence.get('gate_3_admissibility', {}).get('sigma', '?')}; "
            f"reduce spatial instability or improve compatibility"
        )
    elif gate.decision == "REPAIR":
        steps.append(
            "Gate 4 REPAIR -- grounding kappa_L below threshold; "
            "check modality-specific signal quality"
        )

    # Diagnostic warnings
    if diagnostics:
        for w in diagnostics.warnings:
            if "SPATIAL_BLEED" in w:
                steps.append(
                    "Leakage > threshold -- run leave-one-block-out "
                    "to isolate bleeding feature group"
                )
            elif "COLLAPSE_RISK" in w:
                steps.append(
                    "Warmup solver near collapse -- verify target has variance "
                    "and features have signal in this scenario"
                )
            elif "LOW_COHERENCE" in w:
                steps.append(
                    "Low fold coherence in warmup -- folds disagree; "
                    "check spatial split validity"
                )
            elif "WEAK_SOLVER" in w:
                steps.append(
                    "Weak solver discrimination -- model can barely separate; "
                    "consider stronger features or different solver"
                )
            elif "UNEXPLAINED_SPATIAL_PATTERN" in w:
                steps.append(
                    "High residual spatial autocorrelation -- "
                    "model misses spatial structure; consider spatial lag features"
                )

    # Trajectory warnings
    if trajectory:
        if trajectory.convergence_state == "prematurely_stalled":
            steps.append(
                "Prematurely stalled -- uplift < 1% at low alpha/kappa; "
                "current feature ladder insufficient"
            )
        if trajectory.alpha_trend == "regressing":
            steps.append(
                "Alpha regressing -- latest level made quality worse; "
                "check for introduced noise or overfitting"
            )
        if trajectory.sigma_trend == "destabilizing":
            steps.append(
                "Sigma destabilizing -- instability increasing across levels; "
                "added features may introduce spatial heterogeneity"
            )

    return steps


def build_health_cards(
    snapshot: WorkflowSnapshot,
    preset: ThresholdPreset,
    preset_is_override: bool = False,
) -> list[HealthCard]:
    """Build HealthCards for all cells across all levels.

    Args:
        snapshot: Loaded workflow data.
        preset: Threshold preset for gate replay.
        preset_is_override: If True, admission mismatches are
            relabeled as what-if deltas.

    Returns:
        List of HealthCards, one per (cell, level).
    """
    cards: list[HealthCard] = []
    cells = get_cells(snapshot)

    for scenario, target in cells:
        cell = CellKey(scenario=scenario, target=target)

        # Collect per-level certs for trajectory
        certs_by_level: dict[str, dict[str, Any]] = {}
        for level in sorted(snapshot.certificates.keys()):
            cert = get_cert_for_cell(snapshot, scenario, target, level)
            if cert:
                certs_by_level[level] = cert

        # Trajectory (computed once per cell, shared across level cards)
        money_row = get_money_for_cell(snapshot, scenario, target)
        trajectory = analyze_trajectory(
            certs_by_level, money_row, snapshot.provenance, preset
        )

        # Routing (computed once per cell)
        admission_rows = get_admission_for_cell(snapshot, scenario, target)
        routing_row = get_routing_for_cell(snapshot, scenario, target)

        # Gearbox (computed once per cell)
        gearbox = get_gearbox_for_cell(snapshot, scenario, target)

        for level in sorted(certs_by_level.keys()):
            cert = certs_by_level[level]

            # Layer 1: Gate triage
            gate = evaluate_gates(cert, preset)

            # Layer 2: Degradation
            degradation = classify_degradation(cert, gate.decision, preset=preset)

            # Layer 3: Diagnostics
            diag_data = get_diag_for_cell(snapshot, scenario, target, level)
            diagnostics = analyze_diagnostics(diag_data, gearbox, preset)

            # Layer 5: Routing
            routing = analyze_routing(gate.decision, admission_rows, routing_row)
            if preset_is_override and routing:
                routing.conflicts = [
                    c.replace("MISMATCH", "WHAT_IF_DELTA")
                    for c in routing.conflicts
                ]

            # Classify and compose
            health = _classify_health(gate, diagnostics, trajectory, routing)
            next_steps = _generate_next_steps(gate, diagnostics, trajectory, routing)

            conflicts: list[str] = []
            if routing:
                conflicts.extend(routing.conflicts)

            cards.append(HealthCard(
                cell=cell,
                level=level,
                gate=gate,
                degradation=degradation,
                diagnostics=diagnostics,
                trajectory=trajectory,
                routing=routing,
                overall_health=health,
                next_steps=next_steps,
                conflicts=conflicts,
            ))

    return cards
