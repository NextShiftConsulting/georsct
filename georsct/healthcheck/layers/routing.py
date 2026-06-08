"""Layer 5: DGM routing + conflict detection."""

from __future__ import annotations

from typing import Any

from ..models import RoutingResult


def analyze_routing(
    gate_decision: str,
    admission_rows: list[dict[str, Any]],
    routing_row: dict[str, Any] | None,
) -> RoutingResult | None:
    """Detect conflicts between gate replay, admission table, and DGM routing.

    Args:
        gate_decision: EnforcementDecision from Layer 1 replay.
        admission_rows: Admission table rows for this cell (may be empty).
        routing_row: DGM routing table entry for this cell (may be None).

    Returns:
        RoutingResult, or None if no routing or admission data.
    """
    if not admission_rows and routing_row is None:
        return None

    conflicts: list[str] = []
    admission_match: bool | None = None
    internal_decision: str | None = None
    public_decision: str | None = None
    recommended_arm: str | None = None

    # Check admission table vs replay
    if admission_rows:
        # Use the first non-block-level admission row, or the first row
        admission_decision = admission_rows[0].get("enforcement_decision")
        if admission_decision:
            admission_match = (admission_decision == gate_decision)
            if not admission_match:
                conflicts.append(
                    f"ADMISSION_REPLAY_MISMATCH: replay={gate_decision}, "
                    f"admission={admission_decision} at "
                    f"{admission_rows[0].get('gate_reached', 'unknown')}"
                )

    # Check DGM routing vs replay
    if routing_row is not None:
        internal_decision = routing_row.get("internal_decision")
        public_decision = routing_row.get("public_decision")
        recommended_arm = routing_row.get("recommended_arm")

        if internal_decision and internal_decision != gate_decision:
            if gate_decision == "EXECUTE" and internal_decision in ("RE_ENCODE", "REPAIR"):
                conflicts.append(
                    f"ROUTING_CONFLICT: gate says EXECUTE but DGM routed "
                    f"{internal_decision} (over-conservative routing)"
                )
            elif gate_decision == "REPAIR" and internal_decision == "REJECT":
                conflicts.append(
                    f"ROUTING_CONFLICT: gate says REPAIR but DGM routed "
                    f"REJECT (escalation without recovery attempt)"
                )
            else:
                conflicts.append(
                    f"ROUTING_CONFLICT: gate={gate_decision}, "
                    f"DGM={internal_decision}"
                )

    return RoutingResult(
        internal_decision=internal_decision,
        public_decision=public_decision,
        recommended_arm=recommended_arm,
        conflicts=conflicts,
        admission_replay_match=admission_match,
    )
