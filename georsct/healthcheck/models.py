"""Core data models for the diagnostic tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Decision-string contract (protects dgm routing, health_card branches,
# serialization — all from one check at construction time).
#
# The landmine: str(EnforcementDecision.EXECUTE) == "EnforcementDecision.EXECUTE"
# which silently misses every dict key and == comparison. .value is safe;
# str() is the bomb. coerce_decision catches it at GateResult construction
# so no caller ever sees a non-canonical string.
# ---------------------------------------------------------------------------

CANONICAL_DECISIONS: frozenset[str] = frozenset({
    "EXECUTE", "REJECT", "BLOCK", "RE_ENCODE", "REPAIR", "WARN", "FALLBACK",
})


class DecisionContractError(ValueError):
    """A GateResult.decision that is not a bare canonical string."""


def coerce_decision(value: Any) -> str:
    """Coerce any decision-ish value to a bare canonical string, or raise.

    Safe inputs:
        EnforcementDecision.EXECUTE  -> "EXECUTE"   (via .value)
        "EXECUTE"                    -> "EXECUTE"

    Rejected inputs:
        "EnforcementDecision.EXECUTE" -> RAISE (the str(enum) bomb)
        None / unknown token          -> RAISE
    """
    val = getattr(value, "value", value)
    if not isinstance(val, str):
        raise DecisionContractError(
            f"decision must resolve to str, got {type(val).__name__}: {val!r}"
        )
    if val not in CANONICAL_DECISIONS:
        raise DecisionContractError(
            f"decision {val!r} not in canonical vocabulary "
            f"{sorted(CANONICAL_DECISIONS)}. If this looks like "
            f"'EnforcementDecision.EXECUTE', someone used str(enum) "
            f"instead of enum.value."
        )
    return val


@dataclass(frozen=True)
class CellKey:
    scenario: str
    target: str

    def __str__(self) -> str:
        return f"{self.scenario} / {self.target}"


# ---------------------------------------------------------------------------
# Layer outputs
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Layer 1: gate replay (authoritative).

    decision is validated at construction to be a bare canonical string.
    A raw enum, str(enum), or unknown token raises DecisionContractError
    here — loudly, once — instead of silently misrouting downstream.
    """
    decision: str
    gate_reached: str
    gate_evidence: dict[str, Any]
    sub_signal: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", coerce_decision(self.decision))


@dataclass
class DegradationResult:
    """Layer 2: DegradationType classification."""
    degradation_type: str
    confidence: float
    explanation: str
    alpha_level: str
    kappa_level: str


@dataclass
class DiagnosticResult:
    """Layer 3: ratio analysis."""
    leakage: float | None = None
    transfer: float | None = None
    solver: float | None = None
    residual_spatial: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrajectoryResult:
    """Layer 4: cross-level trajectory."""
    levels_present: list[str]
    alpha_trend: str
    sigma_trend: str
    uplift_pct: dict[str, float]
    convergence_state: str
    warnings: list[str] = field(default_factory=list)
    provenance_valid: bool = True


@dataclass
class RoutingResult:
    """Layer 5: DGM routing + conflict detection."""
    internal_decision: str | None = None
    public_decision: str | None = None
    recommended_arm: str | None = None
    conflicts: list[str] = field(default_factory=list)
    admission_replay_match: bool | None = None


# ---------------------------------------------------------------------------
# HealthCard
# ---------------------------------------------------------------------------

@dataclass
class HealthCard:
    """Composed per-cell, per-level health assessment."""
    cell: CellKey
    level: str
    gate: GateResult
    degradation: DegradationResult
    diagnostics: DiagnosticResult | None = None
    trajectory: TrajectoryResult | None = None
    routing: RoutingResult | None = None
    overall_health: str = "healthy"
    next_steps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cell": {"scenario": self.cell.scenario, "target": self.cell.target},
            "level": self.level,
            "overall_health": self.overall_health,
            "gate": {
                "decision": self.gate.decision,
                "gate_reached": self.gate_reached,
                "gate_evidence": self.gate.gate_evidence,
                "sub_signal": self.gate.sub_signal,
            },
            "degradation": {
                "degradation_type": self.degradation.degradation_type,
                "confidence": self.degradation.confidence,
                "explanation": self.degradation.explanation,
                "alpha_level": self.degradation.alpha_level,
                "kappa_level": self.degradation.kappa_level,
            },
        }
        if self.diagnostics is not None:
            result["diagnostics"] = {
                "leakage": self.diagnostics.leakage,
                "transfer": self.diagnostics.transfer,
                "solver": self.diagnostics.solver,
                "residual_spatial": self.diagnostics.residual_spatial,
                "warnings": self.diagnostics.warnings,
            }
        if self.trajectory is not None:
            result["trajectory"] = {
                "levels_present": self.trajectory.levels_present,
                "alpha_trend": self.trajectory.alpha_trend,
                "sigma_trend": self.trajectory.sigma_trend,
                "uplift_pct": self.trajectory.uplift_pct,
                "convergence_state": self.trajectory.convergence_state,
                "provenance_valid": self.trajectory.provenance_valid,
                "warnings": self.trajectory.warnings,
            }
        if self.routing is not None:
            result["routing"] = {
                "internal_decision": self.routing.internal_decision,
                "public_decision": self.routing.public_decision,
                "recommended_arm": self.routing.recommended_arm,
                "conflicts": self.routing.conflicts,
                "admission_replay_match": self.routing.admission_replay_match,
            }
        result["next_steps"] = self.next_steps
        result["conflicts"] = self.conflicts
        return result

    @property
    def gate_reached(self) -> str:
        return self.gate.gate_reached


# ---------------------------------------------------------------------------
# WorkflowSnapshot
# ---------------------------------------------------------------------------

@dataclass
class WorkflowSnapshot:
    """All available data for one experiment run."""
    certificates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    admission_table: list[dict[str, Any]] | None = None
    routing_table: dict[str, Any] | None = None
    gearbox: list[dict[str, Any]] | None = None
    money_table: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    loader_warnings: list[str] = field(default_factory=list)
