"""Tests for GeoRSCT-X execution harness, gearbox, and scoring.

Uses a mock evaluator and demo experts — no external dependencies.
"""

from __future__ import annotations

from typing import Any

import pytest

from georsct.contracts.task_contract import NumericField, TaskContract
from georsct.contracts.task_gold import GoldField, TaskGold
from georsct.evaluation.scoring import (
    score_geocert,
    score_outcome,
    score_process,
)
from georsct.execution.gearbox import rank_experts, select_gear
from georsct.execution.harness import GeoRSCTHarness
from georsct.experts.base import ExpertResult, SpatialExpert
from georsct.experts.hwm_reliability import HWMObservationExpert
from georsct.experts.jrc_surface_water import JRCSurfaceWaterExpert
from georsct.provenance.trace import (
    ExecutionCertificate,
    Gate,
    Step,
    Trace,
    Verdict,
    Weakness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_contract(
    task_id: str = "test-001",
    geometry: str = "prediction",
) -> TaskContract:
    return TaskContract(
        task_id=task_id,
        geometry=geometry,
        reasoning_level=1,
        question="Test question",
        scenario={"event": "test", "region": "test"},
        output_fields=(
            NumericField("loss", abs_tol=1000.0, rel_tol=0.10),
        ),
    )


def _make_gold(task_id: str = "test-001") -> TaskGold:
    return TaskGold(
        task_id=task_id,
        fields=(GoldField("loss", value=10000.0, abs_tol=1000.0, rel_tol=0.10),),
    )


def _make_cert(
    kappa_geom: float = 0.4,
    N: float = 0.6,
    residual_moran: float = 0.2,
    leakage: float = 0.05,
    verdict: Verdict = Verdict.WARN,
) -> ExecutionCertificate:
    return ExecutionCertificate(
        geometry="prediction",
        R=1 - N - 0.1,
        S_sup=0.1,
        N=N,
        kappa_geom=kappa_geom,
        kappa_req=0.7,
        leakage_score=leakage,
        fold_stability=0.9,
        residual_moran=residual_moran,
        verdict=verdict,
    )


def _mock_evaluator_factory(
    initial_kappa: float = 0.4,
    enriched_kappa: float = 0.64,
    initial_N: float = 0.6,
    enriched_N: float = 0.25,
):
    """Create a mock evaluator that improves on enrichment."""
    call_count = [0]

    def evaluator(contract: TaskContract, state: dict[str, Any]) -> ExecutionCertificate:
        feats = state.get("features", {})
        enriched = "surface_water_persistence" in feats
        coverage = feats.get("hwm_coverage", 0.10)
        call_count[0] += 1

        kappa = enriched_kappa if enriched else initial_kappa
        kappa += 0.06 if coverage > 0.3 else 0.0
        N = enriched_N if coverage > 0.3 else initial_N
        moran = 0.20 if enriched else 0.28
        kreq = 0.7

        verdict = Verdict.PASS if kappa >= kreq and N < 0.5 else Verdict.WARN
        gates = [Gate("Grounding", kreq, kappa, kappa >= kreq)]
        return ExecutionCertificate(
            geometry=contract.geometry,
            R=1 - N - 0.1, S_sup=0.1, N=N,
            kappa_geom=round(kappa, 3), kappa_req=kreq,
            leakage_score=0.05, fold_stability=0.91,
            residual_moran=moran, gates=gates, verdict=verdict,
        )

    return evaluator


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestTaskContract:

    def test_valid_contract(self):
        c = _make_contract()
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
        c = _make_contract()
        g = _make_gold()
        # Contract has no 'value' field
        assert not hasattr(c.output_fields[0], "value")
        # Gold has the value
        assert g.fields[0].value == 10000.0


# ---------------------------------------------------------------------------
# Certificate and weakness tests
# ---------------------------------------------------------------------------

class TestExecutionCertificate:

    def test_weakness_vector_low_coverage(self):
        cert = _make_cert(N=0.6)
        wv = cert.weakness_vector()
        types = [w.weakness_type for w in wv]
        assert "low_target_coverage" in types

    def test_weakness_vector_under_supported(self):
        cert = _make_cert(kappa_geom=0.4, N=0.3)
        wv = cert.weakness_vector()
        types = [w.weakness_type for w in wv]
        assert "under_supported_geometry" in types

    def test_weakness_vector_none_when_healthy(self):
        cert = _make_cert(kappa_geom=0.8, N=0.2, residual_moran=0.1, leakage=0.05)
        wv = cert.weakness_vector()
        assert len(wv) == 0

    def test_weakness_vector_capped_at_3(self):
        cert = _make_cert(kappa_geom=0.3, N=0.7, residual_moran=0.5, leakage=0.4)
        wv = cert.weakness_vector(max_weaknesses=3)
        assert len(wv) <= 3

    def test_too_many_weaknesses_fails(self):
        cert = _make_cert(kappa_geom=0.3, N=0.7, residual_moran=0.5, leakage=0.4)
        cert.weakness_vector(max_weaknesses=3)
        assert cert.verdict == Verdict.FAIL

    def test_weakness_ranked_by_severity(self):
        cert = _make_cert(kappa_geom=0.3, N=0.7, residual_moran=0.5)
        wv = cert.weakness_vector()
        severities = [w.severity for w in wv]
        assert severities == sorted(severities, reverse=True)

    def test_is_admissible(self):
        assert _make_cert(verdict=Verdict.PASS).is_admissible()
        assert _make_cert(verdict=Verdict.WARN).is_admissible()
        assert not _make_cert(verdict=Verdict.FAIL).is_admissible()
        assert not _make_cert(verdict=Verdict.SUPPRESS).is_admissible()


# ---------------------------------------------------------------------------
# Gearbox tests
# ---------------------------------------------------------------------------

class TestGearbox:

    def test_select_gear_observe(self):
        cert = _make_cert(N=0.6, kappa_geom=0.8)
        assert select_gear(cert) == "G1_observe"

    def test_select_gear_enrich(self):
        cert = _make_cert(N=0.3, kappa_geom=0.4)
        assert select_gear(cert) == "G2_enrich"

    def test_select_gear_base_when_healthy(self):
        cert = _make_cert(kappa_geom=0.8, N=0.2, residual_moran=0.1)
        assert select_gear(cert) == "G0_base"

    def test_rank_experts_by_delta(self):
        cert = _make_cert(kappa_geom=0.4, N=0.3)
        contract = _make_contract()
        experts = [HWMObservationExpert(), JRCSurfaceWaterExpert()]
        ranked = rank_experts(experts, cert, contract)
        # JRC has higher expected_delta (0.18 vs 0.06)
        assert ranked[0].expert_id == "jrc_surface_water"

    def test_rank_excludes_already_run(self):
        cert = _make_cert(kappa_geom=0.4, N=0.3)
        contract = _make_contract()
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
        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=_mock_evaluator_factory(),
        )
        trace = harness.run(contract)

        assert trace.task_id == "test-001"
        assert len(trace.steps) > 0
        assert trace.certificate is not None

    def test_experts_activated_in_order(self):
        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=_mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        ids = [s.tool_name for s in trace.steps]
        # Primary weakness: under_supported_geometry (severity 0.3 > 0.1)
        # JRC addresses this with higher expected_delta (0.18 vs 0.06)
        assert ids[0] == "jrc_surface_water"

    def test_admission_reason_recorded(self):
        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=_mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        for step in trace.steps:
            assert step.admission_reason != ""
            assert ":" in step.admission_reason

    def test_certificate_before_after_recorded(self):
        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=_mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        for step in trace.steps:
            assert step.certificate_before is not None
            assert step.certificate_after is not None

    def test_muon_rollback_on_regression(self):
        """If an expert worsens kappa, the harness suppresses."""
        def regressing_evaluator(contract, state):
            feats = state.get("features", {})
            # Expert makes kappa WORSE
            kappa = 0.3 if "surface_water_persistence" in feats else 0.5
            return ExecutionCertificate(
                geometry=contract.geometry,
                R=0.5, S_sup=0.2, N=0.3,
                kappa_geom=kappa, kappa_req=0.7,
                leakage_score=0.05, fold_stability=0.9,
                residual_moran=0.2, verdict=Verdict.WARN,
            )

        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[JRCSurfaceWaterExpert()],
            evaluator=regressing_evaluator,
        )
        trace = harness.run(contract)
        assert trace.certificate.verdict == Verdict.SUPPRESS

    def test_max_iters_respected(self):
        contract = _make_contract()

        class InfiniteExpert(SpatialExpert):
            expert_id = "infinite"
            tool_group = "test"
            admissible_geometries = frozenset({"prediction"})
            addresses = frozenset({"under_supported_geometry"})

            def run(self, contract, state):
                return ExpertResult(features={f"f_{id(self)}": 1.0})

        # Always returns under-supported
        def stuck_evaluator(contract, state):
            return _make_cert(kappa_geom=0.4, N=0.3)

        harness = GeoRSCTHarness(
            experts=[InfiniteExpert()],
            evaluator=stuck_evaluator,
            max_iters=3,
        )
        trace = harness.run(contract)
        # Should stop after 1 iteration (expert already_run after first)
        assert len(trace.steps) <= 3

    def test_trace_serializable(self):
        contract = _make_contract()
        harness = GeoRSCTHarness(
            experts=[HWMObservationExpert(), JRCSurfaceWaterExpert()],
            evaluator=_mock_evaluator_factory(),
        )
        trace = harness.run(contract)
        j = trace.to_json()
        assert "test-001" in j
        assert "admission_reason" in j


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
        # Pred matches gold exactly
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
        assert score.tool_acc == 1.0  # right tools
        assert score.order_score < 1.0  # wrong order


class TestOutcomeScoring:

    def test_exact_match(self):
        pred = Trace(task_id="test-001", final_json={"loss": 10000.0})
        gold = _make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 1.0
        assert score.num_score == 1.0

    def test_within_tolerance(self):
        pred = Trace(task_id="test-001", final_json={"loss": 10500.0})
        gold = _make_gold()  # abs_tol=1000, rel_tol=0.10
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 1.0

    def test_outside_tolerance(self):
        pred = Trace(task_id="test-001", final_json={"loss": 20000.0})
        gold = _make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 0.0
        assert score.num_score < 1.0

    def test_missing_field_scores_zero(self):
        pred = Trace(task_id="test-001", final_json={})
        gold = _make_gold()
        score = score_outcome(pred, gold)
        assert score.hit_at_tol == 0.0
        assert score.num_score == 0.0


class TestGeoCertScoring:

    def test_pass_verdict_full_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=_make_cert(verdict=Verdict.PASS),
        )
        outcome = score_outcome(pred, _make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == outcome.num_score * 1.0
        assert not geo.lucky_hit

    def test_fail_verdict_zero_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=_make_cert(verdict=Verdict.FAIL),
        )
        outcome = score_outcome(pred, _make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == 0.0

    def test_lucky_hit_detected(self):
        """Correct answer + failed certificate = lucky hit."""
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=_make_cert(verdict=Verdict.FAIL),
        )
        outcome = score_outcome(pred, _make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.lucky_hit is True
        assert geo.geocert_score == 0.0

    def test_warn_verdict_partial_score(self):
        pred = Trace(
            task_id="test-001",
            final_json={"loss": 10000.0},
            certificate=_make_cert(verdict=Verdict.WARN),
        )
        outcome = score_outcome(pred, _make_gold())
        geo = score_geocert(pred, outcome)
        assert geo.geocert_score == outcome.num_score * 0.8
