"""Tests for GeoRSCT-X harness, gearbox, and scoring.

Uses test fixtures (demo experts, mock evaluators) — no external dependencies.
"""

from __future__ import annotations

import pytest

from georsct.application.gearbox import rank_experts, select_gear
from georsct.application.harness import GeoRSCTHarness
from georsct.contracts.task_contract import TaskContract
from georsct.evaluation.scoring import (
    score_geocert,
    score_outcome,
    score_process,
)
from georsct.ports.spatial_expert import ExpertResult, SpatialExpert
from georsct.provenance.trace import (
    ExecutionCertificate,
    Step,
    Trace,
    Verdict,
)

from tests.fixtures.contracts import make_cert, make_contract, make_gold
from tests.fixtures.evaluators import mock_evaluator_factory
from tests.fixtures.experts import HWMObservationExpert, JRCSurfaceWaterExpert


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestTaskContract:

    def test_valid_contract(self):
        c = make_contract()
        assert c.task_id == "test-001"
        assert c.geometry == "prediction"

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="geometry"):
            TaskContract(
                task_id="x", geometry="invalid",
                reasoning_level=0, question="q",
            )

    def test_invalid_reasoning_level_raises(self):
        with pytest.raises(ValueError, match="reasoning_level"):
            TaskContract(
                task_id="x", geometry="prediction",
                reasoning_level=5, question="q",
            )

    def test_gold_split(self):
        """Gold values are in TaskGold, not TaskContract."""
        c = make_contract()
        g = make_gold()
        assert not hasattr(c.output_fields[0], "value")
        assert g.fields[0].value == 10000.0


# ---------------------------------------------------------------------------
# Certificate and weakness tests
# ---------------------------------------------------------------------------

class TestExecutionCertificate:

    def test_weakness_vector_low_coverage(self):
        cert = make_cert(N=0.6)
        wv = cert.weakness_vector()
        types = [w.weakness_type for w in wv]
        assert "low_target_coverage" in types

    def test_weakness_vector_under_supported(self):
        cert = make_cert(kappa_coupling=0.4, N=0.3)
        wv = cert.weakness_vector()
        types = [w.weakness_type for w in wv]
        assert "under_supported_geometry" in types

    def test_weakness_vector_none_when_healthy(self):
        cert = make_cert(kappa_coupling=0.8, N=0.2, residual_moran=0.1, leakage=0.05)
        wv = cert.weakness_vector()
        assert len(wv) == 0

    def test_weakness_vector_capped_at_3(self):
        cert = make_cert(kappa_coupling=0.3, N=0.7, residual_moran=0.5, leakage=0.4)
        wv = cert.weakness_vector(max_weaknesses=3)
        assert len(wv) <= 3

    def test_weakness_vector_no_longer_sets_fail(self):
        """FAIL is decided by the harness, not weakness_vector."""
        cert = make_cert(kappa_coupling=0.3, N=0.7, residual_moran=0.5, leakage=0.4)
        cert.weakness_vector(max_weaknesses=3)
        assert cert.verdict == Verdict.WARN

    def test_all_weaknesses_returns_full_audit(self):
        cert = make_cert(kappa_coupling=0.3, N=0.7, residual_moran=0.5, leakage=0.4)
        all_w = cert.all_weaknesses()
        assert len(all_w) == 4

    def test_has_high_severity_unresolved(self):
        cert = make_cert(kappa_coupling=0.8, N=0.7, residual_moran=0.1)
        assert cert.has_high_severity_unresolved()
        healthy = make_cert(kappa_coupling=0.8, N=0.2, residual_moran=0.1)
        assert not healthy.has_high_severity_unresolved()

    def test_weakness_ranked_by_severity(self):
        cert = make_cert(kappa_coupling=0.3, N=0.7, residual_moran=0.5)
        wv = cert.weakness_vector()
        severities = [w.severity for w in wv]
        assert severities == sorted(severities, reverse=True)

    def test_is_admissible(self):
        assert make_cert(verdict=Verdict.PASS).is_admissible()
        assert make_cert(verdict=Verdict.WARN).is_admissible()
        assert not make_cert(verdict=Verdict.FAIL).is_admissible()
        assert not make_cert(verdict=Verdict.SUPPRESS).is_admissible()


# ---------------------------------------------------------------------------
# Gearbox tests
# ---------------------------------------------------------------------------

class TestGearbox:

    def test_select_gear_observe(self):
        cert = make_cert(N=0.6, kappa_coupling=0.8)
        assert select_gear(cert) == "G1_observe"

    def test_select_gear_enrich(self):
        cert = make_cert(N=0.3, kappa_coupling=0.4)
        assert select_gear(cert) == "G2_enrich"

    def test_select_gear_base_when_healthy(self):
        cert = make_cert(kappa_coupling=0.8, N=0.2, residual_moran=0.1)
        assert select_gear(cert) == "G0_base"

    def test_rank_experts_by_delta(self):
        cert = make_cert(kappa_coupling=0.4, N=0.3)
        contract = make_contract()
        experts = [HWMObservationExpert(), JRCSurfaceWaterExpert()]
        ranked = rank_experts(experts, cert, contract)
        assert ranked[0].expert_id == "jrc_surface_water"

    def test_rank_excludes_already_run(self):
        cert = make_cert(kappa_coupling=0.4, N=0.3)
        contract = make_contract()
        experts = [JRCSurfaceWaterExpert()]
        ranked = rank_experts(
            experts, cert, contract,
            already_run=frozenset({"jrc_surface_water"}),
        )
        assert len(ranked) == 0


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------

class TestHarness:

    def test_basic_execution(self):
        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        assert trace.task_id == "test-001"
        assert len(trace.steps) > 0
        assert trace.certificate is not None

    def test_experts_activated_in_order(self):
        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        ids = [s.tool_name for s in trace.steps]
        assert ids[0] == "jrc_surface_water"

    def test_admission_reason_recorded(self):
        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        for step in trace.steps:
            assert step.admission_reason != ""
            assert ":" in step.admission_reason

    def test_certificate_before_after_recorded(self):
        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        for step in trace.steps:
            assert step.certificate_before is not None
            assert step.certificate_after is not None

    def test_muon_rollback_on_regression(self):
        """If an expert worsens kappa, the harness suppresses."""
        def regressing_evaluator(contract, state):
            feats = state.get("features", {})
            kappa = 0.3 if "surface_water_persistence" in feats else 0.5
            return ExecutionCertificate(
                geometry=contract.geometry,
                R=0.5, S_sup=0.2, N=0.3,
                kappa_coupling=kappa, kappa_threshold=0.7,
                leakage_score=0.05, fold_stability=0.9,
                residual_moran=0.2, verdict=Verdict.WARN,
            )

        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[JRCSurfaceWaterExpert()],
            evaluator=regressing_evaluator,
        )
        trace = harness.run(contract)
        assert trace.certificate.verdict == Verdict.SUPPRESS

    def test_muon_rollback_reverts_state(self):
        """Muon rollback must revert features so solver sees clean state."""
        solver_features: list[dict] = []

        def capturing_solver(contract, state):
            solver_features.append(dict(state.get("features", {})))
            return {}

        def regressing_evaluator(contract, state):
            feats = state.get("features", {})
            kappa = 0.3 if "surface_water_persistence" in feats else 0.5
            return ExecutionCertificate(
                geometry=contract.geometry,
                R=0.5, S_sup=0.2, N=0.3,
                kappa_coupling=kappa, kappa_threshold=0.7,
                leakage_score=0.05, fold_stability=0.9,
                residual_moran=0.2, verdict=Verdict.WARN,
            )

        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[JRCSurfaceWaterExpert()],
            evaluator=regressing_evaluator,
            solver=capturing_solver,
        )
        harness.run(contract)
        # Solver must NOT see the suppressed expert's features
        assert "surface_water_persistence" not in solver_features[0]

    def test_max_iters_respected(self):
        contract = make_contract()

        class InfiniteExpert(SpatialExpert):
            expert_id = "infinite"
            tool_group = "test"
            admissible_geometries = frozenset({"prediction"})
            addresses = frozenset({"under_supported_geometry"})

            def run(self, contract, state):
                return ExpertResult(features={f"f_{id(self)}": 1.0})

        def stuck_evaluator(contract, state):
            return make_cert(kappa_coupling=0.4, N=0.3)

        harness = GeoRSCTHarness(
            experts=[InfiniteExpert()],
            evaluator=stuck_evaluator,
            max_iters=3,
        )
        trace = harness.run(contract)
        assert len(trace.steps) <= 3

    def test_trace_serializable(self):
        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        j = trace.to_json()
        assert "test-001" in j
        assert "admission_reason" in j

    def test_severity_gate_skips_marginal_weaknesses(self):
        """Marginal weaknesses below severity threshold skip the expert loop."""
        def healthy_evaluator(contract, state):
            return make_cert(kappa_coupling=0.65, N=0.25, residual_moran=0.15)

        contract = make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=healthy_evaluator,
        )
        trace = harness.run(contract)
        assert len(trace.steps) == 0
        assert trace.certificate is not None

    def test_expert_preserves_declared(self):
        """Experts declare spatial invariants they preserve."""
        hwm = HWMObservationExpert()
        jrc = JRCSurfaceWaterExpert()
        assert "topology" in hwm.preserves
        assert "adjacency" in hwm.preserves
        assert "area_proportional" in jrc.preserves


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestProcessScoring:

    def _gold_trace(self) -> Trace:
        return Trace(
            task_id="test-001",
            steps=[
                Step(0, "hwm_observation_reliability", "satellite_eo",
                     {"geometry": "prediction", "gear": "G1_observe"},
                     True, True, {}),
                Step(1, "jrc_surface_water", "reanalysis_environmental",
                     {"geometry": "prediction", "gear": "G2_enrich"},
                     True, True, {}),
            ],
        )

    def test_perfect_process_score(self):
        gold = self._gold_trace()
        pred = self._gold_trace()
        score = score_process(pred, gold)
        assert score.tool_acc == 1.0
        assert score.inst_acc == 1.0
        assert score.tool_use_score > 0.9

    def test_empty_pred_scores_low(self):
        gold = self._gold_trace()
        pred = Trace(task_id="test-001")
        score = score_process(pred, gold)
        assert score.tool_acc == 0.0

    def test_wrong_order_partial_credit(self):
        gold = self._gold_trace()
        pred = Trace(
            task_id="test-001",
            steps=[
                Step(0, "jrc_surface_water", "reanalysis_environmental",
                     {"geometry": "prediction"}, True, True, {}),
                Step(1, "hwm_observation_reliability", "satellite_eo",
                     {"geometry": "prediction"}, True, True, {}),
            ],
        )
        score = score_process(pred, gold)
        assert score.tool_acc == 1.0
        assert score.order_score < 1.0


class TestOutcomeScoring:

    def test_exact_match(self):
        pred = Trace(task_id="test-001", final_json={"loss": 10000.0})
        gold = make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 1.0
        assert score.num_score == 1.0

    def test_within_tolerance(self):
        pred = Trace(task_id="test-001", final_json={"loss": 10500.0})
        gold = make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 1.0

    def test_outside_tolerance(self):
        pred = Trace(task_id="test-001", final_json={"loss": 20000.0})
        gold = make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 0.0
        assert score.num_score < 1.0

    def test_missing_field_scores_zero(self):
        pred = Trace(task_id="test-001", final_json={})
        gold = make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 0.0
        assert score.num_score == 0.0


class TestGeoCertScoring:

    def test_pass_verdict_full_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=make_cert(verdict=Verdict.PASS),
        )
        outcome = score_outcome(pred, make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == outcome.num_score * 1.0
        assert not geo.lucky_hit

    def test_fail_verdict_zero_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=make_cert(verdict=Verdict.FAIL),
        )
        outcome = score_outcome(pred, make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == 0.0

    def test_lucky_hit_detected(self):
        """Correct answer + failed certificate = lucky hit."""
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=make_cert(verdict=Verdict.FAIL),
        )
        outcome = score_outcome(pred, make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.lucky_hit is True
        assert geo.geocert_score == 0.0

    def test_warn_verdict_partial_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=make_cert(verdict=Verdict.WARN),
        )
        outcome = score_outcome(pred, make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == outcome.num_score * 0.8
