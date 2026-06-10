"""Layer 1: Gate replay (authoritative).

Always recomputes gate decisions from thresholds -- never trusts admission table.
"""

from __future__ import annotations

from ..models import GateResult
from ..thresholds import ThresholdPreset


def evaluate_gates(
    cert: dict,
    preset: ThresholdPreset,
) -> GateResult:
    """Replay a certificate through gates using the given preset.

    Gate order: Gate 1 -> Gate 2 -> Gate 3 -> Gate 4.
    Stops at first failure.

    Args:
        cert: Certificate dict with alpha, kappa, sigma, etc.
        preset: Threshold preset for gate evaluation.

    Returns:
        GateResult with decision, gate_reached, evidence, and sub_signal.
    """
    alpha = cert.get("alpha", 0.0)
    # ADR-020 D8.5: kappa_compat only; bare "kappa" is forbidden
    kappa = cert.get("kappa_compat")
    sigma = cert.get("sigma", 0.0)
    coherence = cert.get("coherence")
    kappa_L = cert.get("kappa_L")

    # V-011: Gate 1 reads derived noise_admissibility, falls back to raw N.
    evidence_block = cert.get("evidence", {})
    noise_admissibility = evidence_block.get("noise_admissibility")
    noise_fallback = noise_admissibility is None
    if noise_fallback:
        noise_admissibility = cert.get("N", 1.0)

    evidence: dict = {}

    # --- Gate 1: Integrity Guard (OR logic — canonical SequentialGatekeeper) ---
    # Either low noise OR sufficient alpha is enough to pass.
    gate1_n_pass = noise_admissibility < preset.N_thr
    gate1_alpha_pass = alpha >= preset.alpha_min
    gate1_pass = gate1_n_pass or gate1_alpha_pass

    evidence["gate_1"] = {
        "noise_admissibility": round(noise_admissibility, 6),
        "noise_fallback_to_raw_N": noise_fallback,
        "N_thr": preset.N_thr,
        "N_pass": gate1_n_pass,
        "alpha": round(alpha, 6),
        "alpha_min": preset.alpha_min,
        "alpha_pass": gate1_alpha_pass,
        "pass": gate1_pass,
    }

    if not gate1_pass:
        # Both checks failed — report the dominant failure
        sub = "N_FLOOR_BREACH_AND_ALPHA_LOW"
        return GateResult(
            decision="REJECT",
            gate_reached="GATE_1_INTEGRITY",
            gate_evidence=evidence,
            sub_signal=sub,
        )

    # --- Gate 2: Consensus Gate ---
    # Three canonical paths:
    #   1. coherence present → check against c_min
    #   2. coherence absent + require_coherence=True → fail closed (BLOCK)
    #   3. coherence absent + require_coherence=False → pass (legacy path)
    if coherence is not None:
        gate2_pass = coherence >= preset.c_min
        evidence["gate_2"] = {
            "coherence": round(coherence, 6),
            "c_min": preset.c_min,
            "require_coherence": preset.gate_2_require_coherence,
            "pass": gate2_pass,
        }
        if not gate2_pass:
            return GateResult(
                decision="BLOCK",
                gate_reached="GATE_2_CONSENSUS",
                gate_evidence=evidence,
                sub_signal="COHERENCE_LOW",
            )
    elif preset.gate_2_require_coherence:
        # Fail closed: coherence required but absent
        evidence["gate_2"] = {
            "coherence": None,
            "c_min": preset.c_min,
            "require_coherence": True,
            "pass": False,
            "failure_path": "coherence_absent_fail_closed",
        }
        return GateResult(
            decision="BLOCK",
            gate_reached="GATE_2_CONSENSUS",
            gate_evidence=evidence,
            sub_signal="COHERENCE_ABSENT_FAIL_CLOSED",
        )
    else:
        # Legacy pass: coherence not required
        evidence["gate_2"] = {
            "coherence": None,
            "require_coherence": False,
            "pass": True,
            "failure_path": "coherence_absent_legacy_pass",
        }

    # --- Gate 3: Admissibility (Oobleck) ---
    # ADR-020 D8.5: kappa=None → fail closed with explicit warning
    if kappa is None:
        evidence["gate_3"] = {
            "kappa_compat": None,
            "kappa_source": cert.get("kappa_source", "unavailable"),
            "pass": False,
            "failure_path": "kappa_unavailable_d8_5",
        }
        return GateResult(
            decision="BLOCK",
            gate_reached="GATE_3_ADMISSIBILITY",
            gate_evidence=evidence,
            sub_signal="KAPPA_UNAVAILABLE",
        )

    gate3_sigma_pass = sigma <= preset.sigma_thr
    kappa_req = preset.kappa_req(sigma)
    kappa_margin = kappa - kappa_req
    landauer_zone = abs(kappa_margin) <= preset.epsilon_L

    if not gate3_sigma_pass:
        sub = "KAPPA_BELOW_OOBLECK"
        gate3_pass = False
    elif kappa_margin < -preset.epsilon_L:
        sub = "KAPPA_BELOW_OOBLECK"
        gate3_pass = False
    elif landauer_zone and kappa_margin < 0:
        sub = "KAPPA_LANDAUER_FAIL"
        gate3_pass = False
    else:
        sub = None
        gate3_pass = True

    evidence["gate_3"] = {
        "sigma": round(sigma, 6),
        "sigma_thr": preset.sigma_thr,
        "sigma_pass": gate3_sigma_pass,
        "kappa_compat": round(kappa, 6),
        "kappa_req": round(kappa_req, 6),
        "kappa_margin": round(kappa_margin, 6),
        "kappa_source": cert.get("kappa_source", "unknown"),
        "epsilon_L": preset.epsilon_L,
        "landauer_zone": landauer_zone,
        "pass": gate3_pass,
    }

    if not gate3_pass:
        return GateResult(
            decision="RE_ENCODE",
            gate_reached="GATE_3_ADMISSIBILITY",
            gate_evidence=evidence,
            sub_signal=sub,
        )

    # --- Gate 3B: Prior Calibration (conditional) ---
    if preset.enable_gate_3b:
        r_bar = cert.get("r_bar")
        if r_bar is not None:
            gate3b_pass = r_bar >= preset.r_bar_min
            evidence["gate_3b"] = {
                "r_bar": round(r_bar, 6),
                "r_bar_min": preset.r_bar_min,
                "pass": gate3b_pass,
            }
            if not gate3b_pass:
                return GateResult(
                    decision="RE_ENCODE",
                    gate_reached="GATE_3B_PRIOR_CALIBRATION",
                    gate_evidence=evidence,
                    sub_signal="R_BAR_BELOW_THRESHOLD",
                )
        else:
            evidence["gate_3b"] = {
                "r_bar": None,
                "r_bar_min": preset.r_bar_min,
                "pass": True,
                "note": "r_bar absent, gate 3B not evaluable",
            }

    # --- Gate 4: Grounding Gate ---
    if kappa_L is not None:
        gate4_pass = kappa_L >= preset.kappa_L_min
        evidence["gate_4"] = {
            "kappa_L": round(kappa_L, 6),
            "kappa_L_min": preset.kappa_L_min,
            "pass": gate4_pass,
        }
        if not gate4_pass:
            return GateResult(
                decision="REPAIR",
                gate_reached="GATE_4_GROUNDING",
                gate_evidence=evidence,
                sub_signal="KAPPA_L_BELOW_THRESHOLD",
            )
    else:
        evidence["gate_4"] = {"kappa_L": None, "pass": True, "note": "no kappa_L data"}

    # --- All passed ---
    return GateResult(
        decision="EXECUTE",
        gate_reached="ALL_PASSED",
        gate_evidence=evidence,
    )
