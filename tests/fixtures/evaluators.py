"""Mock certificate evaluators for testing."""

from __future__ import annotations

from typing import Any

from georsct.contracts.task_contract import TaskContract
from georsct.provenance.trace import (
    ExecutionCertificate,
    Gate,
    Verdict,
)


def mock_evaluator_factory(
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
            kappa_coupling=round(kappa, 3), kappa_threshold=kreq,
            leakage_score=0.05, fold_stability=0.91,
            residual_moran=moran, gates=gates, verdict=verdict,
        )

    return evaluator
