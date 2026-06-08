"""Layer 5: Routing and conflict detection tests."""

from georsct.healthcheck.layers.routing import analyze_routing


class TestNoData:
    def test_no_admission_no_routing(self):
        result = analyze_routing("EXECUTE", [], None)
        assert result is None


class TestAdmissionReplay:
    def test_match(self):
        admission = [{"enforcement_decision": "EXECUTE", "gate_reached": "ALL_PASSED"}]
        result = analyze_routing("EXECUTE", admission, None)
        assert result.admission_replay_match is True
        assert len(result.conflicts) == 0

    def test_mismatch(self):
        admission = [{"enforcement_decision": "REJECT", "gate_reached": "GATE_1_INTEGRITY"}]
        result = analyze_routing("EXECUTE", admission, None)
        assert result.admission_replay_match is False
        assert any("ADMISSION_REPLAY_MISMATCH" in c for c in result.conflicts)


class TestRoutingConflict:
    def test_over_conservative(self):
        routing = {
            "internal_decision": "RE_ENCODE",
            "public_decision": "RE_ENCODE",
            "recommended_arm": "arm_b",
        }
        result = analyze_routing("EXECUTE", [], routing)
        assert any("over-conservative" in c for c in result.conflicts)

    def test_escalation_without_recovery(self):
        routing = {
            "internal_decision": "REJECT",
            "public_decision": "REJECT",
        }
        result = analyze_routing("REPAIR", [], routing)
        assert any("escalation without recovery" in c for c in result.conflicts)

    def test_generic_mismatch(self):
        routing = {"internal_decision": "BLOCK"}
        result = analyze_routing("RE_ENCODE", [], routing)
        assert any("ROUTING_CONFLICT" in c for c in result.conflicts)

    def test_no_conflict_when_matching(self):
        routing = {"internal_decision": "EXECUTE"}
        result = analyze_routing("EXECUTE", [], routing)
        assert len(result.conflicts) == 0

    def test_recommended_arm_captured(self):
        routing = {
            "internal_decision": "EXECUTE",
            "recommended_arm": "arm_a",
        }
        result = analyze_routing("EXECUTE", [], routing)
        assert result.recommended_arm == "arm_a"


class TestCombined:
    def test_admission_and_routing_conflicts(self):
        admission = [{"enforcement_decision": "REJECT", "gate_reached": "GATE_1"}]
        routing = {"internal_decision": "RE_ENCODE"}
        result = analyze_routing("EXECUTE", admission, routing)
        assert result.admission_replay_match is False
        assert len(result.conflicts) == 2
