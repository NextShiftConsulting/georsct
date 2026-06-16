"""GeoRSCT-X execution harness — Layer 2.

ReAct loop: plan -> admit -> execute -> observe -> recertify.

The harness executes benchmark tasks by activating spatial experts,
recording admissibility provenance, and emitting certificates.

The key differentiator from TerraBench:
    The trace is an admissibility log, not just a tool-use log.

Import rule: depends on contracts, provenance, experts.base (ABCs),
and execution.gearbox. NEVER imports domain types directly --
the CertificateEvaluator is injected as a Protocol callable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

from georsct.contracts.task_contract import TaskContract
from georsct.execution.gearbox import rank_experts, select_gear
from georsct.experts.base import SpatialExpert
from georsct.provenance.trace import (
    ExecutionCertificate,
    Step,
    Trace,
    Verdict,
)


# ---------------------------------------------------------------------------
# CertificateEvaluator protocol (injected, not imported from domain)
# ---------------------------------------------------------------------------

class CertificateEvaluator(Protocol):
    """Protocol for certificate evaluation.

    Injected at composition root. The harness never imports domain
    certificate types directly -- this protocol is the boundary.

    Implementors typically wrap:
      - georsct.domain.construct_certificate (measurement)
      - georsct.application.use_cases.gate_3b_decision (control)
      - georsct.domain.kappa (geometry compatibility)
    """

    def __call__(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> ExecutionCertificate: ...


# ---------------------------------------------------------------------------
# Solver protocol (injected, not imported)
# ---------------------------------------------------------------------------

class Solver(Protocol):
    """Protocol for the final answer solver.

    Injected at composition root. Produces the numeric output from
    the enriched state. Production: /pair-score or TreeMeasurementAdapter.
    """

    def __call__(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Default solver (pass-through from state)
# ---------------------------------------------------------------------------

def _default_solver(
    contract: TaskContract,
    state: dict[str, Any],
) -> dict[str, float]:
    """Pass-through solver: reads pred_{key} from features."""
    features = state.get("features", {})
    out: dict[str, float] = {}
    for f in contract.output_fields:
        out[f.key] = float(features.get(f"pred_{f.key}", 0.0))
    return out


# ---------------------------------------------------------------------------
# GeoRSCT-X Harness
# ---------------------------------------------------------------------------

# Convergence threshold for kappa_coupling fixpoint.
_KAPPA_EPS = 1e-3


class GeoRSCTHarness:
    """Executable certificate-governed workflow harness.

    Combines three paper lessons:
      - TerraBench-style executable traces
      - DMoE-style modular experts
      - Muon-style admissibility checks

    The harness runs a ReAct loop:
      1. Evaluate base certificate
      2. Gearbox diagnoses weakness
      3. Rank and admit one spatial expert
      4. Execute expert
      5. Re-evaluate certificate
      6. Suppress if compatibility worsens (Muon rollback)
      7. Repeat until PASS / WARN / ESCALATE / SUPPRESS

    Args:
        experts: Registered spatial experts (injected).
        evaluator: Certificate evaluator (injected protocol).
        solver: Final answer solver (injected protocol, optional).
        max_iters: Maximum enrichment iterations.
    """

    def __init__(
        self,
        experts: list[SpatialExpert],
        evaluator: CertificateEvaluator,
        solver: Solver | None = None,
        max_iters: int = 6,
    ):
        self.experts = experts
        self.evaluate = evaluator
        self.solve = solver or _default_solver
        self.max_iters = max_iters

    def run(self, contract: TaskContract) -> Trace:
        """Execute one benchmark task end-to-end.

        Returns a Trace with admissibility provenance, certificate
        trajectory, artifacts, and final numeric output.
        """
        trace = Trace(task_id=contract.task_id)
        state: dict[str, Any] = {
            "scenario": dict(contract.scenario),
            "features": {},
        }
        artifacts: dict[str, Any] = {}

        # Initial certificate evaluation
        cert = self.evaluate(contract, state)
        admitted: set[str] = set()

        for _ in range(self.max_iters):
            gear = select_gear(cert)

            # G0: representation sufficient. R: suppress/escalate.
            if gear == "G0_base":
                break
            if gear == "R_reverse":
                cert.verdict = Verdict.SUPPRESS
                break

            # Rank and admit best expert
            ranked = rank_experts(
                self.experts, cert, contract, frozenset(admitted),
            )
            if not ranked:
                # No admissible expert left for this weakness.
                # FAIL if high-severity weaknesses remain unresolved;
                # otherwise ESCALATE for human review.
                if cert.has_high_severity_unresolved():
                    cert.verdict = Verdict.FAIL
                elif cert.primary_weakness() != "none":
                    if cert.verdict == Verdict.WARN:
                        cert.verdict = Verdict.ESCALATE
                break

            expert = ranked[0]
            admitted.add(expert.expert_id)

            # Snapshot certificate before expert activation
            cert_before = asdict(cert)

            # Execute expert
            result = expert.run(contract, state)
            state["features"].update(result.features)
            for art in result.artifacts:
                art.created_by_step = len(trace.steps)
                artifacts[art.artifact_id] = art

            # Re-evaluate certificate
            cert_after = self.evaluate(contract, state)

            # Record step with admissibility provenance
            trace.steps.append(Step(
                index=len(trace.steps),
                tool_name=expert.expert_id,
                tool_group=expert.tool_group,
                args={"geometry": contract.geometry, "gear": gear},
                valid=True,
                success=True,
                observation={
                    "compatibility_delta": result.compatibility_delta,
                },
                artifact_ids=[a.artifact_id for a in result.artifacts],
                admission_reason=f"{gear}:{cert.primary_weakness()}",
                gear=gear,
                certificate_before=cert_before,
                certificate_after=asdict(cert_after),
                compatibility_delta=cert_after.kappa_coupling - cert.kappa_coupling,
            ))

            # Muon admissibility check: rollback if expert hurt the certificate
            if (
                cert_after.kappa_coupling < cert.kappa_coupling
                or cert_after.residual_moran > cert.residual_moran
            ):
                cert_after.verdict = Verdict.SUPPRESS
                cert = cert_after
                break

            # Fixpoint: if enrichment no longer moves compatibility, stop
            if abs(cert_after.kappa_coupling - cert.kappa_coupling) < _KAPPA_EPS:
                cert = cert_after
                break

            cert = cert_after

        trace.certificate = cert
        trace.final_json = self.solve(contract, state)
        return trace
