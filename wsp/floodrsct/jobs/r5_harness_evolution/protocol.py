"""Evolution protocol for R5 harness evolution.

Runs the step loop: solve -> score -> attribute failures -> propose patch
-> validate -> accept/reject -> emit certificate trajectory.

This module contains no model-specific logic. VLM calls and evolver calls
are injected as callables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .apply_patch import PatchApplicationError, apply_patch
from .attribution import build_failure_report
from .harness_schema import (
    CertificateTrajectory,
    HarnessPatch,
    HarnessVersion,
    RsctBlock,
    SpatialDiagnostics,
    TrajectoryStep,
)
from .harness_store import HarnessStore
from .scoring import (
    JudgmentMetrics,
    aggregate_judgments,
    simplex_from_judgments,
    zone_macro_f1,
)
from .validators import validate_patch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acceptance rule
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceDecision:
    accepted: bool
    reason: str
    primary_delta: float = 0.0
    schema_validity_delta: float = 0.0


def default_acceptance_rule(
    baseline_score: float,
    candidate_score: float,
    baseline_judgments: JudgmentMetrics,
    candidate_judgments: JudgmentMetrics,
    min_delta: float = 0.0,
) -> AcceptanceDecision:
    """Accept if primary metric non-decreasing and schema validity holds."""
    primary_delta = candidate_score - baseline_score
    schema_delta = (
        candidate_judgments.schema_validity_rate
        - baseline_judgments.schema_validity_rate
    )
    claim_delta = (
        candidate_judgments.unprovided_claim_rate
        - baseline_judgments.unprovided_claim_rate
    )

    if primary_delta < min_delta:
        return AcceptanceDecision(
            accepted=False,
            reason=f"primary metric delta {primary_delta:.4f} < {min_delta}",
            primary_delta=primary_delta,
            schema_validity_delta=schema_delta,
        )

    if schema_delta < -0.01:
        return AcceptanceDecision(
            accepted=False,
            reason=f"schema validity regressed by {schema_delta:.4f}",
            primary_delta=primary_delta,
            schema_validity_delta=schema_delta,
        )

    if claim_delta > 0.01:
        return AcceptanceDecision(
            accepted=False,
            reason=f"unprovided claim rate increased by {claim_delta:.4f}",
            primary_delta=primary_delta,
            schema_validity_delta=schema_delta,
        )

    return AcceptanceDecision(
        accepted=True,
        reason="all gates passed",
        primary_delta=primary_delta,
        schema_validity_delta=schema_delta,
    )


# ---------------------------------------------------------------------------
# RSCT block builder (ADR-020 D8.4 provenance)
# ---------------------------------------------------------------------------

def build_rsct_block(
    simplex: dict | None,
    kappa_req_value: float | None = None,
) -> RsctBlock:
    """Build an RsctBlock from available simplex signals.

    Args:
        simplex: dict with R, S_sup, N keys, or None if unavailable.
        kappa_req_value: precomputed kappa_req(sigma) threshold, or None.

    Returns:
        RsctBlock with provenance fields populated per ADR-020 D8.4.
    """
    warnings: list[str] = []

    if simplex is None:
        warnings.append("WARN_RSN_UNAVAILABLE")
        warnings.append("WARN_KAPPA_UNAVAILABLE")
        return RsctBlock(warnings=warnings)

    R = simplex.get("R")
    S_sup = simplex.get("S_sup")
    N = simplex.get("N")

    if R is None or N is None:
        warnings.append("WARN_KAPPA_COMPAT_UNAVAILABLE")
        return RsctBlock(R=R, S_sup=S_sup, N=N, warnings=warnings)

    kappa_compat = R * (1.0 - N)
    alpha = R / (R + N) if (R + N) > 0 else None
    passed = None
    if kappa_req_value is not None:
        passed = kappa_compat >= kappa_req_value

    # kappa_difficulty unavailable for classification tasks (D8.2)
    warnings.append("WARN_KAPPA_DIFFICULTY_UNAVAILABLE_FOR_CLASSIFICATION")

    return RsctBlock(
        R=R,
        S_sup=S_sup,
        N=N,
        alpha=alpha,
        kappa_compat=kappa_compat,
        kappa_difficulty=None,
        active_metric="kappa_compat",
        kappa_source="rsn_simplex",
        kappa_formula="R*(1-N)",
        kappa_authority="canonical_simplex_estimate",
        sigma=simplex.get("sigma"),
        kappa_req=kappa_req_value,
        passed=passed,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Protocol config
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    experiment_id: str
    agent_vlm: str
    evolver_model: str
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    heldout_ids: set[str] = field(default_factory=set)
    n_steps: int = 3
    batch_size: int = 50
    primary_metric: str = "zone_macro_f1"
    min_accept_delta: float = 0.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_evolution(
    config: EvolutionConfig,
    initial_harness: HarnessVersion,
    store: HarnessStore,
    run_vlm_batch: Callable,
    judge_batch: Callable,
    load_references: Callable,
    propose_patch: Callable,
) -> CertificateTrajectory:
    """Run the full evolution protocol.

    Args:
        config: experiment configuration
        initial_harness: H_0
        store: artifact storage
        run_vlm_batch: (agent_vlm, harness, zcta_ids) -> list[dict]
        judge_batch: (outputs) -> list[dict]  (activation/adherence)
        load_references: (zcta_ids) -> list[dict]
        propose_patch: (harness, failure_report, step) -> HarnessPatch

    Returns:
        CertificateTrajectory with all steps recorded.
    """
    trajectory = CertificateTrajectory(
        experiment_id=config.experiment_id,
        agent_vlm=config.agent_vlm,
        evolver_model=config.evolver_model,
        primary_metric=config.primary_metric,
    )

    H = initial_harness
    store.save_harness(H.harness_id, H.to_dict())

    # Baseline step
    log.info("Step 0: baseline evaluation")
    baseline_outputs = run_vlm_batch(config.agent_vlm, H, config.train_ids)
    baseline_judgments = judge_batch(baseline_outputs)
    baseline_refs = load_references(config.train_ids)
    baseline_score = zone_macro_f1(baseline_outputs, baseline_refs)
    baseline_metrics = aggregate_judgments(baseline_judgments)
    current_simplex = simplex_from_judgments(baseline_metrics)

    trajectory.append_step(TrajectoryStep(
        step=0,
        harness_id=H.harness_id,
        split="train",
        primary_score=baseline_score,
        activation_rate=baseline_metrics.activation_rate,
        adherence_rate=baseline_metrics.adherence_rate,
        grounding_rate=baseline_metrics.grounding_rate,
        schema_validity_rate=baseline_metrics.schema_validity_rate,
        unprovided_claim_rate=baseline_metrics.unprovided_claim_rate,
        rsct=build_rsct_block(simplex=current_simplex),
        patch_decision="baseline",
        n_zctas=len(config.train_ids),
    ))

    current_score = baseline_score
    current_metrics = baseline_metrics

    # Evolution steps
    for t in range(1, config.n_steps + 1):
        log.info("Step %d / %d", t, config.n_steps)

        # 1. Build failure report
        failure_report = build_failure_report(
            predictions=baseline_outputs,
            references=baseline_refs,
            judgments=baseline_judgments,
            step=t,
            harness_id=H.harness_id,
            batch_id=f"batch_{t:03d}",
            primary_metric_value=current_score,
            primary_metric_baseline=baseline_score,
            heldout_ids=config.heldout_ids,
        )
        store.save_failure_report(t, failure_report)

        # 2. Evolver proposes patch
        patch = propose_patch(H, failure_report, t)
        store.save_patch(patch.patch_id, patch.to_dict())

        # 3. Validate patch
        validation = validate_patch(
            patch, H, heldout_ids=config.heldout_ids
        )
        if not validation.passed:
            log.warning(
                "Step %d: patch rejected by validator: %s",
                t, validation.errors,
            )
            trajectory.append_step(TrajectoryStep(
                step=t,
                harness_id=H.harness_id,
                split="train",
                primary_score=current_score,
                activation_rate=current_metrics.activation_rate,
                adherence_rate=current_metrics.adherence_rate,
                grounding_rate=current_metrics.grounding_rate,
                schema_validity_rate=current_metrics.schema_validity_rate,
                unprovided_claim_rate=current_metrics.unprovided_claim_rate,
                rsct=build_rsct_block(simplex=current_simplex),
                patch_decision="rejected_validation",
                n_zctas=len(config.train_ids),
            ))
            continue

        # 4. Apply patch
        try:
            H_candidate = apply_patch(H, patch)
        except PatchApplicationError as exc:
            log.warning("Step %d: patch application failed: %s", t, exc)
            trajectory.append_step(TrajectoryStep(
                step=t,
                harness_id=H.harness_id,
                split="train",
                primary_score=current_score,
                rsct=build_rsct_block(simplex=current_simplex),
                patch_decision="rejected_application_error",
                n_zctas=len(config.train_ids),
            ))
            continue

        # 5. Evaluate candidate on validation set
        val_outputs = run_vlm_batch(
            config.agent_vlm, H_candidate, config.validation_ids
        )
        val_judgments = judge_batch(val_outputs)
        val_refs = load_references(config.validation_ids)
        val_score = zone_macro_f1(val_outputs, val_refs)
        val_metrics = aggregate_judgments(val_judgments)

        # 6. Accept or reject
        val_simplex = simplex_from_judgments(val_metrics)
        decision = default_acceptance_rule(
            baseline_score=current_score,
            candidate_score=val_score,
            baseline_judgments=current_metrics,
            candidate_judgments=val_metrics,
            min_delta=config.min_accept_delta,
        )

        if decision.accepted:
            log.info(
                "Step %d: ACCEPTED (delta=%.4f, reason=%s)",
                t, decision.primary_delta, decision.reason,
            )
            H = H_candidate
            store.save_harness(H.harness_id, H.to_dict())
            current_score = val_score
            current_metrics = val_metrics
            current_simplex = val_simplex

            # Re-run on train to update baseline for next step
            baseline_outputs = run_vlm_batch(
                config.agent_vlm, H, config.train_ids
            )
            baseline_judgments = judge_batch(baseline_outputs)
        else:
            log.info(
                "Step %d: REJECTED (delta=%.4f, reason=%s)",
                t, decision.primary_delta, decision.reason,
            )

        trajectory.append_step(TrajectoryStep(
            step=t,
            harness_id=H.harness_id,
            split="validation",
            primary_score=val_score,
            activation_rate=val_metrics.activation_rate,
            adherence_rate=val_metrics.adherence_rate,
            grounding_rate=val_metrics.grounding_rate,
            schema_validity_rate=val_metrics.schema_validity_rate,
            unprovided_claim_rate=val_metrics.unprovided_claim_rate,
            rsct=build_rsct_block(simplex=val_simplex),
            patch_decision="accepted" if decision.accepted else "rejected",
            n_zctas=len(config.validation_ids),
        ))

    # Final test evaluation
    log.info("Final evaluation on test set")
    test_outputs = run_vlm_batch(config.agent_vlm, H, config.test_ids)
    test_judgments = judge_batch(test_outputs)
    test_refs = load_references(config.test_ids)
    test_score = zone_macro_f1(test_outputs, test_refs)
    test_metrics = aggregate_judgments(test_judgments)

    test_simplex = simplex_from_judgments(test_metrics)
    trajectory.append_step(TrajectoryStep(
        step=config.n_steps + 1,
        harness_id=H.harness_id,
        split="test",
        primary_score=test_score,
        activation_rate=test_metrics.activation_rate,
        adherence_rate=test_metrics.adherence_rate,
        grounding_rate=test_metrics.grounding_rate,
        schema_validity_rate=test_metrics.schema_validity_rate,
        unprovided_claim_rate=test_metrics.unprovided_claim_rate,
        rsct=build_rsct_block(simplex=test_simplex),
        patch_decision="final_test",
        n_zctas=len(config.test_ids),
    ))

    store.save_trajectory(trajectory.to_dict())
    log.info(
        "Evolution complete: %d steps, baseline=%.4f, final_test=%.4f",
        config.n_steps, baseline_score, test_score,
    )

    return trajectory
