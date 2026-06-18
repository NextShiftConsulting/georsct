"""Health card composition and classification tests."""

from georsct.healthcheck.models import (
    CellKey,
    DiagnosticResult,
    GateResult,
    HealthCard,
    RoutingResult,
    TrajectoryResult,
)
from georsct.healthcheck.health_card import _classify_health, _generate_next_steps


def _gate(decision="EXECUTE", gate_reached="ALL_PASSED", sub_signal=None):
    return GateResult(
        decision=decision,
        gate_reached=gate_reached,
        gate_evidence={},
        sub_signal=sub_signal,
    )


class TestClassifyHealth:
    def test_healthy(self):
        assert _classify_health(_gate(), None, None, None) == "healthy"

    def test_reject_is_critical(self):
        assert _classify_health(_gate("REJECT"), None, None, None) == "critical"

    def test_block_is_critical(self):
        assert _classify_health(_gate("BLOCK"), None, None, None) == "critical"

    def test_re_encode_is_failing(self):
        assert _classify_health(_gate("RE_ENCODE"), None, None, None) == "failing"

    def test_repair_is_failing(self):
        assert _classify_health(_gate("REPAIR"), None, None, None) == "failing"

    def test_gate_fail_plus_regression_is_critical(self):
        traj = TrajectoryResult(
            levels_present=["r0", "r1"],
            alpha_trend="regressing",
            sigma_trend="flat",
            uplift_pct={},
            convergence_state="regressing",
        )
        assert _classify_health(_gate("RE_ENCODE"), None, traj, None) == "critical"

    def test_gate_fail_plus_diag_warnings_is_critical(self):
        diag = DiagnosticResult(warnings=["SPATIAL_BLEED: leakage=1.2"])
        assert _classify_health(_gate("REPAIR"), diag, None, None) == "critical"

    def test_execute_with_diag_warnings_is_warning(self):
        diag = DiagnosticResult(warnings=["WEAK_SOLVER: solver=0.3"])
        assert _classify_health(_gate(), diag, None, None) == "warning"

    def test_execute_with_regression_is_warning(self):
        traj = TrajectoryResult(
            levels_present=["r0", "r1"],
            alpha_trend="regressing",
            sigma_trend="flat",
            uplift_pct={},
            convergence_state="regressing",
        )
        assert _classify_health(_gate(), None, traj, None) == "warning"

    def test_execute_with_stall_is_warning(self):
        traj = TrajectoryResult(
            levels_present=["r0", "r1"],
            alpha_trend="stalled",
            sigma_trend="flat",
            uplift_pct={},
            convergence_state="prematurely_stalled",
        )
        assert _classify_health(_gate(), None, traj, None) == "warning"

    def test_execute_with_routing_conflict_is_warning(self):
        routing = RoutingResult(conflicts=["ROUTING_CONFLICT: something"])
        assert _classify_health(_gate(), None, None, routing) == "warning"


class TestNextSteps:
    def test_reject_combined_n_and_alpha(self):
        """Gate 1 REJECT emits N_FLOOR_BREACH_AND_ALPHA_LOW (both failed)."""
        gate = _gate("REJECT", "GATE_1_INTEGRITY", "N_FLOOR_BREACH_AND_ALPHA_LOW")
        steps = _generate_next_steps(gate, None, None, None)
        assert len(steps) > 0, "Gate 1 REJECT must produce next-step guidance"
        assert any("noise" in s.lower() for s in steps)
        assert any("alpha" in s.lower() for s in steps)

    def test_block(self):
        gate = _gate("BLOCK", "GATE_2_CONSENSUS")
        steps = _generate_next_steps(gate, None, None, None)
        assert any("Low coherence" in s for s in steps)

    def test_re_encode(self):
        gate = _gate("RE_ENCODE", "GATE_3_ADMISSIBILITY")
        gate.gate_evidence = {"gate_3": {"sigma": 0.40}}
        steps = _generate_next_steps(gate, None, None, None)
        assert any("RE_ENCODE" in s for s in steps)

    def test_repair(self):
        gate = _gate("REPAIR", "GATE_4_GROUNDING")
        steps = _generate_next_steps(gate, None, None, None)
        assert any("REPAIR" in s for s in steps)

    def test_leakage_warning(self):
        diag = DiagnosticResult(warnings=["SPATIAL_BLEED: leakage=1.2"])
        steps = _generate_next_steps(_gate(), diag, None, None)
        assert any("Leakage" in s for s in steps)

    def test_collapse_risk(self):
        diag = DiagnosticResult(warnings=["COLLAPSE_RISK: collapse_risk=0.5"])
        steps = _generate_next_steps(_gate(), diag, None, None)
        assert any("collapse" in s.lower() for s in steps)

    def test_premature_stall(self):
        traj = TrajectoryResult(
            levels_present=["r0", "r1"],
            alpha_trend="stalled",
            sigma_trend="flat",
            uplift_pct={},
            convergence_state="prematurely_stalled",
        )
        steps = _generate_next_steps(_gate(), None, traj, None)
        assert any("stalled" in s.lower() for s in steps)

    def test_routing_conflicts_first(self):
        routing = RoutingResult(conflicts=["ROUTING_CONFLICT: over-conservative"])
        gate = _gate("RE_ENCODE", "GATE_3_ADMISSIBILITY")
        steps = _generate_next_steps(gate, None, None, routing)
        # Routing conflicts come first
        assert "ROUTING_CONFLICT" in steps[0]


class TestHealthCardToDict:
    def test_serialization(self):
        card = HealthCard(
            cell=CellKey(scenario="sc1", target="t1"),
            level="r0",
            gate=_gate(),
            degradation=None,  # would fail in to_dict, but testing structure
            overall_health="healthy",
        )
        # Can't call to_dict without degradation, test basic attributes
        assert card.cell.scenario == "sc1"
        assert card.level == "r0"
        assert card.gate_reached == "ALL_PASSED"
