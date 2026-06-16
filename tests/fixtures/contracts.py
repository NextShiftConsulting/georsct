"""Synthetic contracts and gold values for testing."""

from __future__ import annotations

from georsct.contracts.task_contract import NumericField, TaskContract
from georsct.contracts.task_gold import GoldField, TaskGold
from georsct.provenance.trace import ExecutionCertificate, Verdict


def make_contract(
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


def make_gold(task_id: str = "test-001") -> TaskGold:
    return TaskGold(
        task_id=task_id,
        fields=(GoldField("loss", value=10000.0, abs_tol=1000.0, rel_tol=0.10),),
    )


def make_cert(
    kappa_coupling: float = 0.4,
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
        kappa_coupling=kappa_coupling,
        kappa_threshold=0.7,
        leakage_score=leakage,
        fold_stability=0.9,
        residual_moran=residual_moran,
        verdict=verdict,
    )
