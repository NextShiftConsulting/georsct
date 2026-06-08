"""Render HealthCards as text or JSON."""

from __future__ import annotations

import json
from typing import Any

from .models import CellKey, HealthCard


HEALTH_LABELS = {
    "healthy": "HEALTHY",
    "warning": "WARNING",
    "failing": "FAILING",
    "critical": "CRITICAL",
}


def _render_card_text(card: HealthCard, explain: bool = False) -> str:
    """Render a single HealthCard as text."""
    lines: list[str] = []
    label = HEALTH_LABELS.get(card.overall_health, card.overall_health.upper())
    lines.append(f"  [{label}] {card.cell} (level={card.level})")

    # Gate
    if card.gate.decision == "EXECUTE":
        lines.append("  Gate:        EXECUTE (all gates passed)")
    else:
        sub = f", sub_signal={card.gate.sub_signal}" if card.gate.sub_signal else ""
        lines.append(
            f"  Gate:        {card.gate.decision} at {card.gate.gate_reached}{sub}"
        )

    # Degradation
    lines.append(
        f"  Degradation: {card.degradation.degradation_type} "
        f"({card.degradation.explanation})"
    )

    # Diagnostics
    if card.diagnostics:
        parts: list[str] = []
        if card.diagnostics.leakage is not None:
            flag = " [!]" if any("SPATIAL_BLEED" in w for w in card.diagnostics.warnings) else ""
            parts.append(f"leakage={card.diagnostics.leakage:.3f}{flag}")
        if card.diagnostics.solver is not None:
            flag = " [!]" if any("WEAK_SOLVER" in w for w in card.diagnostics.warnings) else ""
            parts.append(f"solver={card.diagnostics.solver:.3f}{flag}")
        if card.diagnostics.residual_spatial is not None:
            flag = " [!]" if any("UNEXPLAINED" in w for w in card.diagnostics.warnings) else ""
            parts.append(f"residual_spatial={card.diagnostics.residual_spatial:.3f}{flag}")
        for w in card.diagnostics.warnings:
            if "COLLAPSE_RISK" in w or "LOW_COHERENCE" in w:
                parts.append(f"[!] {w.split(': ', 1)[-1]}")
        if parts:
            lines.append(f"  Diagnostics: {', '.join(parts)}")

    # Trajectory
    if card.trajectory:
        t = card.trajectory
        if not t.provenance_valid:
            lines.append("  Trajectory:  SKIPPED (provenance mismatch)")
        else:
            lines.append(f"  Trajectory:  alpha {t.alpha_trend}, sigma {t.sigma_trend}")
            if t.uplift_pct:
                uplift_str = ", ".join(
                    f"{k} {v:+.1f}%" for k, v in t.uplift_pct.items()
                )
                lines.append(f"               uplift: {uplift_str}")
            lines.append(f"               state: {t.convergence_state}")

    # Routing
    if card.routing:
        if card.routing.conflicts:
            for c in card.routing.conflicts:
                lines.append(f"  Routing:     [!] {c}")
        elif card.routing.internal_decision:
            lines.append(f"  Routing:     DGM agrees ({card.routing.internal_decision})")

    # Next steps
    if card.next_steps:
        lines.append("  Next steps:")
        for i, step in enumerate(card.next_steps, 1):
            lines.append(f"    {i}. {step}")

    # Explain mode: show gate math
    if explain and card.gate.gate_evidence:
        lines.append("  Gate evidence:")
        for gate_name, ev in card.gate.gate_evidence.items():
            lines.append(f"    {gate_name}: {ev}")

    return "\n".join(lines)


def render_text(
    cards: list[HealthCard],
    metadata: dict[str, Any],
    explain: bool = False,
    warnings_only: bool = False,
    summary_only: bool = False,
) -> str:
    """Render full text report."""
    lines: list[str] = []

    experiment = metadata.get("experiment", "unknown")
    preset = metadata.get("preset", "unknown")
    preset_source = metadata.get("preset_source", "certificate")

    # Count health states (use latest level per cell)
    cells_latest: dict[str, HealthCard] = {}
    for card in cards:
        key = str(card.cell)
        if key not in cells_latest or card.level > cells_latest[key].level:
            cells_latest[key] = card

    counts = {"healthy": 0, "warning": 0, "failing": 0, "critical": 0}
    for card in cells_latest.values():
        counts[card.overall_health] = counts.get(card.overall_health, 0) + 1

    lines.append("Workflow Health Report")
    lines.append("=" * 50)
    lines.append(f"Experiment: {experiment}")
    lines.append(f"Preset:     {preset} ({preset_source})")
    lines.append(
        f"Cells:      {len(cells_latest)}  |  "
        f"Healthy: {counts['healthy']}  |  "
        f"Warning: {counts['warning']}  |  "
        f"Failing: {counts['failing']}  |  "
        f"Critical: {counts['critical']}"
    )
    lines.append("")

    if summary_only:
        lines.append(_render_summary(cards))
        return "\n".join(lines)

    # Sort: critical first, then failing, warning, healthy
    priority = {"critical": 0, "failing": 1, "warning": 2, "healthy": 3}
    display_cards = sorted(cards, key=lambda c: (priority.get(c.overall_health, 4), str(c.cell), c.level))

    if warnings_only:
        display_cards = [c for c in display_cards if c.overall_health != "healthy"]

    for card in display_cards:
        lines.append(_render_card_text(card, explain=explain))
        lines.append("")

    lines.append("=" * 50)
    lines.append("SUMMARY")
    lines.append("=" * 50)
    lines.append(_render_summary(cards))

    return "\n".join(lines)


def _render_summary(cards: list[HealthCard]) -> str:
    """Aggregate top issues across all cards."""
    issues: list[str] = []

    # Count patterns
    gate_failures: dict[str, int] = {}
    diag_patterns: dict[str, int] = {}
    stalls = 0
    regressions = 0
    routing_conflicts = 0
    provenance_mismatches = 0

    for card in cards:
        if card.gate.decision != "EXECUTE":
            key = f"{card.gate.decision} at {card.gate.gate_reached}"
            gate_failures[key] = gate_failures.get(key, 0) + 1

        if card.diagnostics:
            for w in card.diagnostics.warnings:
                tag = w.split(":")[0]
                diag_patterns[tag] = diag_patterns.get(tag, 0) + 1

        if card.trajectory:
            if card.trajectory.convergence_state == "prematurely_stalled":
                stalls += 1
            if card.trajectory.alpha_trend == "regressing":
                regressions += 1
            if not card.trajectory.provenance_valid:
                provenance_mismatches += 1

        if card.routing:
            routing_conflicts += len(card.routing.conflicts)

    for key, count in gate_failures.items():
        issues.append(f"{count} cell(s) at {key}")
    for tag, count in diag_patterns.items():
        issues.append(f"{count} cell(s) with {tag}")
    if stalls:
        issues.append(f"{stalls} cell(s) prematurely stalled")
    if regressions:
        issues.append(f"{regressions} cell(s) with trajectory regression")
    if routing_conflicts:
        issues.append(f"{routing_conflicts} routing conflict(s)")
    if provenance_mismatches:
        issues.append(f"{provenance_mismatches} cell(s) with provenance mismatch")

    if not issues:
        return "No issues detected."

    lines = ["Top issues:"]
    for issue in issues:
        lines.append(f"  - {issue}")
    return "\n".join(lines)


def render_json(
    cards: list[HealthCard],
    metadata: dict[str, Any],
) -> str:
    """Render full JSON report."""
    # Group by cell
    cell_groups: dict[str, dict[str, Any]] = {}
    for card in cards:
        key = str(card.cell)
        if key not in cell_groups:
            cell_groups[key] = {
                "cell": {"scenario": card.cell.scenario, "target": card.cell.target},
                "levels": [],
                "trajectory": None,
                "overall_health": "healthy",
            }
        cell_groups[key]["levels"].append({
            "level": card.level,
            "overall_health": card.overall_health,
            "gate": {
                "decision": card.gate.decision,
                "gate_reached": card.gate.gate_reached,
                "gate_evidence": card.gate.gate_evidence,
                "sub_signal": card.gate.sub_signal,
            },
            "degradation": {
                "degradation_type": card.degradation.degradation_type,
                "confidence": card.degradation.confidence,
                "explanation": card.degradation.explanation,
                "alpha_level": card.degradation.alpha_level,
                "kappa_level": card.degradation.kappa_level,
            },
            "diagnostics": {
                "leakage": card.diagnostics.leakage,
                "transfer": card.diagnostics.transfer,
                "solver": card.diagnostics.solver,
                "residual_spatial": card.diagnostics.residual_spatial,
                "warnings": card.diagnostics.warnings,
            } if card.diagnostics else None,
            "routing": {
                "internal_decision": card.routing.internal_decision,
                "public_decision": card.routing.public_decision,
                "conflicts": card.routing.conflicts,
                "admission_replay_match": card.routing.admission_replay_match,
            } if card.routing else None,
            "next_steps": card.next_steps,
            "conflicts": card.conflicts,
        })
        if card.trajectory and cell_groups[key]["trajectory"] is None:
            cell_groups[key]["trajectory"] = {
                "levels_present": card.trajectory.levels_present,
                "alpha_trend": card.trajectory.alpha_trend,
                "sigma_trend": card.trajectory.sigma_trend,
                "uplift_pct": card.trajectory.uplift_pct,
                "convergence_state": card.trajectory.convergence_state,
                "provenance_valid": card.trajectory.provenance_valid,
                "warnings": card.trajectory.warnings,
            }

    # Set cell-level health to worst level health
    priority = {"critical": 0, "failing": 1, "warning": 2, "healthy": 3}
    for group in cell_groups.values():
        worst = min(
            (l["overall_health"] for l in group["levels"]),
            key=lambda h: priority.get(h, 4),
        )
        group["overall_health"] = worst

    # Counts
    counts = {"healthy": 0, "warning": 0, "failing": 0, "critical": 0}
    for group in cell_groups.values():
        counts[group["overall_health"]] = counts.get(group["overall_health"], 0) + 1

    report = {
        "experiment": metadata.get("experiment", "unknown"),
        "preset": metadata.get("preset", "unknown"),
        "preset_source": metadata.get("preset_source", "certificate"),
        "summary": {
            "n_cells": len(cell_groups),
            **counts,
        },
        "cells": list(cell_groups.values()),
    }

    return json.dumps(report, indent=2, default=str)
