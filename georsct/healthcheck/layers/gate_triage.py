"""Layer 1: Gate replay (authoritative).

Delegates to yrsn-controlplane's SequentialGatekeeper (ADR-064).
This module is the SINGLE bridging point between georsct's certificate
dicts and the canonical gatekeeper.  All vocabulary (gate_evidence keys,
sub_signal strings) is the controlplane's vocabulary — no translation.
"""

from __future__ import annotations

from yrsn_controlplane import (
    CPGatekeeperInput,
    GatekeeperConfig,
    SequentialGatekeeper,
)

from ..models import GateResult
from ..thresholds import ThresholdPreset


# ---------------------------------------------------------------------------
# ThresholdPreset -> GatekeeperConfig mapping
#
# Two renames (oobleck_steepness -> steepness, oobleck_sigma_c -> sigma_c),
# four controlplane-only defaults (gate_3b_action, enable_task_residual_floor_gate,
# enable_conditioning_filter, enable_contract_coverage_gate).
# Diagnostic-only fields (leakage_warn, solver_warn) have no gate equivalent.
# ---------------------------------------------------------------------------

def _config_from_preset(preset: ThresholdPreset) -> GatekeeperConfig:
    """Map a ThresholdPreset to a GatekeeperConfig."""
    return GatekeeperConfig(
        N_thr=preset.N_thr,
        alpha_min=preset.alpha_min,
        c_min=preset.c_min,
        gate_2_require_coherence=preset.gate_2_require_coherence,
        sigma_thr=preset.sigma_thr,
        kappa_base=preset.kappa_base,
        lambda_turbulence=preset.lambda_turbulence,
        steepness=preset.oobleck_steepness,
        sigma_c=preset.oobleck_sigma_c,
        epsilon_L=preset.epsilon_L,
        landauer_sigma_tiebreaker=preset.landauer_sigma_tiebreaker,
        enable_gate_3b=preset.enable_gate_3b,
        r_bar_min=preset.r_bar_min,
        kappa_L_min=preset.kappa_L_min,
    )


def _input_from_cert(cert: dict) -> CPGatekeeperInput:
    """Build a CPGatekeeperInput from a certificate dict.

    Cert dicts use 'kappa_compat' (ADR-020 D8.5).  Raw N is passed
    via the evidence dict so Gate 1's noise_admissibility fallback works.
    Coherence is passed via evidence so Gate 2 reads it.
    """
    alpha = cert.get("alpha", 0.0)
    kappa = cert.get("kappa_compat", 0.0)
    sigma = cert.get("sigma", 0.0)

    # Build evidence dict — Gate 1 reads noise_admissibility, Gate 2 reads coherence
    evidence: dict = {}
    evidence_block = cert.get("evidence", {})

    # V-011: prefer derived noise_admissibility, fall back to raw N
    noise_admissibility = evidence_block.get("noise_admissibility")
    if noise_admissibility is not None:
        evidence["noise_admissibility"] = noise_admissibility
    evidence["N"] = cert.get("N", 1.0)

    # Gate 2 reads coherence from evidence
    coherence = cert.get("coherence")
    if coherence is not None:
        evidence["coherence"] = coherence

    evidence["R"] = cert.get("R", 0.0)
    evidence["S_sup"] = cert.get("S_sup", 0.0)
    evidence["kappa_source"] = cert.get("kappa_source", "unknown")

    return CPGatekeeperInput(
        alpha=alpha,
        kappa_compat=kappa,
        sigma=sigma,
        source_mode="direct",
        evidence=evidence,
        kappa_L=cert.get("kappa_L"),
    )


def evaluate_gates(
    cert: dict,
    preset: ThresholdPreset,
) -> GateResult:
    """Replay a certificate through gates using the given preset.

    Delegates to SequentialGatekeeper (yrsn-controlplane).  Returns
    GateResult with controlplane vocabulary for gate_evidence keys
    and sub_signal strings.

    Args:
        cert: Certificate dict with alpha, kappa, sigma, etc.
        preset: Threshold preset for gate evaluation.

    Returns:
        GateResult with decision, gate_reached, evidence, and sub_signal.
    """
    # Pre-check: kappa_compat=None cannot be passed to CPGatekeeperInput
    # (it's a required float). Return BLOCK directly for this edge case.
    kappa = cert.get("kappa_compat")
    if kappa is None:
        return GateResult(
            decision="BLOCK",
            gate_reached="GATE_3_ADMISSIBILITY",
            gate_evidence={
                "gate_3_admissibility": {
                    "kappa_compat": None,
                    "kappa_source": cert.get("kappa_source", "unavailable"),
                    "status": "failed",
                    "failure_path": "kappa_unavailable",
                },
            },
            sub_signal="kappa_unavailable",
        )

    config = _config_from_preset(preset)
    gk = SequentialGatekeeper(config)
    gk_input = _input_from_cert(cert)
    result = gk.evaluate(gk_input)

    # Translate GatekeeperResult -> GateResult
    # decision: enum -> bare string via coerce_decision in GateResult.__post_init__
    # gate_reached: enum -> bare string via .value
    # gate_evidence: passed through (controlplane vocabulary)
    # sub_signal: failure_reason string (controlplane vocabulary)
    return GateResult(
        decision=result.decision,  # coerce_decision handles enum -> str
        gate_reached=result.gate_reached.value,
        gate_evidence=result.gate_evidence,
        sub_signal=result.failure_reason,
    )
