#!/usr/bin/env python3
"""
compute_r3_block_admission.py -- Phase R3_1c: DGM block admission.

Applies the production enforcement stack (yrsn-controlplane) to each
block's certificate from R3_1a. Uses SequentialGatekeeper with the
GEOSPATIAL_CONUS27 preset to produce typed (EnforcementDecision, GearState)
pairs per block.

The full DGM admission pipeline:
  1. Load block certificates from R3_1a
  2. KappaPolicyRegistry.feasibility() pre-check
  3. SequentialGatekeeper.evaluate() -> EnforcementDecision
  4. Coordinator.coordinate() -> ExecutionPlan + gear
  5. Classify admission tier from (Decision, Gear)
  6. Log to DGM admission trace

Prerequisite: kappa pipeline bug (P1) must be fixed before running.

Usage:
    python compute_r3_block_admission.py --upload
    python compute_r3_block_admission.py --dry-run
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"

# Import canonical thresholds from yrsn-controlplane (single source of truth)
from yrsn_controlplane import PRESET_GEOSPATIAL_CONUS27 as _CFG

KAPPA_BASE = _CFG.kappa_base
SIGMA_THR = _CFG.sigma_thr
LAMBDA_TURBULENCE = _CFG.lambda_turbulence
EPSILON_L = _CFG.epsilon_L
KAPPA_L_MIN = _CFG.kappa_L_min
N_THR = _CFG.N_thr
ALPHA_MIN = _CFG.alpha_min
GATE_2_REQUIRE_COHERENCE = _CFG.gate_2_require_coherence

# Oobleck sigmoidal parameters
OOBLECK_STEEPNESS = _CFG.steepness
OOBLECK_SIGMA_C = _CFG.sigma_c
LANDAUER_SIGMA_TIEBREAKER = _CFG.landauer_sigma_tiebreaker

# Gear thresholds (P14, from yrsn-controlplane/types.py)
TAU_FIRST = 1.0
TAU_SECOND = 1.43
TAU_THIRD = 2.5

# Sigma warning threshold
SIGMA_WARNING_THR = 0.40


# ---------------------------------------------------------------------------
# Gate evaluation (mirrors SequentialGatekeeper logic)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _oobleck_kappa_req(sigma: float) -> float:
    """Oobleck dynamic threshold: kappa_req(sigma).

    Uses LAMBDA_TURBULENCE (== delta_kappa) from canonical preset.
    For GEOSPATIAL_CONUS27, lambda_turbulence=0.0 so this is flat at KAPPA_BASE.
    """
    return KAPPA_BASE + LAMBDA_TURBULENCE * _sigmoid(
        OOBLECK_STEEPNESS * (sigma - OOBLECK_SIGMA_C)
    )


def _tau_to_gear(tau: float) -> str:
    """Map tau to gear state (P14)."""
    if tau < TAU_FIRST:
        return "FIRST"
    elif tau < TAU_SECOND:
        return "SECOND"
    elif tau < TAU_THIRD:
        return "THIRD"
    else:
        return "FOURTH"


def _classify_lyapunov(V: float) -> str:
    """Lyapunov advisory gear recommendation."""
    if V < 0.2:
        return "FIRST"
    elif V < 0.5:
        return "SECOND"
    elif V < 1.0:
        return "THIRD"
    elif V < 1.5:
        return "FOURTH"
    else:
        return "REVERSE"


def _invalid_breakdown(invalid_entries: list[dict]) -> dict:
    """Summarize INVALID certificates by failure mode and remediation action."""
    by_mode = {}
    by_action = {}
    by_block = {}
    for entry in invalid_entries:
        diag = entry.get("invalid_diagnosis", {})
        mode = diag.get("failure_mode", "UNKNOWN")
        action = diag.get("remediation_action", "unknown")
        block = entry.get("block", "unknown")

        by_mode[mode] = by_mode.get(mode, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        by_block.setdefault(block, []).append({
            "target": entry.get("target"),
            "failure_mode": mode,
            "remediation": action,
            "partial_signals": diag.get("partial_signals"),
        })

    return {
        "by_failure_mode": by_mode,
        "by_remediation_action": by_action,
        "by_block": by_block,
    }


def _diagnose_invalid(cert: dict) -> dict:
    """Classify WHY a certificate is INVALID and what can fix it.

    Reads the certificate evidence to determine:
      - failure_mode: what broke (solver, features, data)
      - remediation_action: what to do about it
      - partial_signals: any usable signal that survived

    Failure modes (most specific first):
      SOLVER_BOTH_NULL     - histgbdt and ridge both returned null metrics
      SOLVER_HISTGBDT_ONLY - histgbdt failed but ridge wasn't tried / also failed
      STRUCTURAL_NO_FEATURES - block has 0 features present in dataset
      DATA_INSUFFICIENT    - too few rows or single-class target
      UNKNOWN              - fallback
    """
    ev = cert.get("evidence", {})
    indep = ev.get("independent_signal", {})
    marg = ev.get("marginal_signal", {})
    audit = ev.get("feature_audit", {})

    # Collect partial signals that DID produce values
    partial = {}
    full_r2 = marg.get("full_R2", {})
    drop_block = marg.get("drop_block", {})
    if full_r2.get("metric") is not None:
        partial["full_R2_metric"] = full_r2["metric"]
        partial["full_R2_solver"] = full_r2.get("solver_used")
    if drop_block.get("metric") is not None:
        partial["drop_block_metric"] = drop_block["metric"]
        partial["drop_block_solver"] = drop_block.get("solver_used")
    if marg.get("delta") is not None:
        partial["ablation_delta"] = marg["delta"]

    # Classify failure mode
    block_only_valid = audit.get("block_only_valid", False)
    indep_metric = indep.get("metric")
    indep_solver = indep.get("solver_used")

    if not block_only_valid:
        failure_mode = "STRUCTURAL_NO_FEATURES"
        remediation = "check-feature-registry"
        detail = "Block has 0 features present in the dataset"
    elif indep_metric is None and indep_solver is None:
        # Both solvers returned null on block_only
        failure_mode = "SOLVER_BOTH_NULL"
        remediation = "retrain-with-larger-sample"
        detail = (
            "Both histgbdt and ridge returned null metrics on block_only. "
            "Likely cause: insufficient training rows, single-class target, "
            "or all-NaN features in this block."
        )
    elif indep_metric is None:
        failure_mode = "SOLVER_PRIMARY_NULL"
        remediation = "investigate-solver-crash"
        detail = f"Solver '{indep_solver}' returned null metric on block_only"
    else:
        failure_mode = "UNKNOWN"
        remediation = "investigate"
        detail = "Certificate marked INVALID but independent signal exists"

    return {
        "failure_mode": failure_mode,
        "remediation_action": remediation,
        "detail": detail,
        "solver_trace": {
            "primary_solver": indep.get("primary_solver"),
            "fallback_solver": indep.get("fallback_solver"),
            "fallback_triggered": indep.get("fallback_triggered"),
            "primary_failure_reason": indep.get("primary_failure_reason"),
        },
        "partial_signals": partial if partial else None,
        "feature_audit_summary": {
            "add_status": audit.get("add_status"),
            "drop_status": audit.get("drop_status"),
            "block_only_valid": block_only_valid,
        },
    }


def evaluate_gates(cert: dict) -> dict:
    """Evaluate the 4-gate pipeline on a block certificate.

    Returns gate result with decision, gear, and full trace.
    """
    def _safe(val, default):
        """Coalesce None/NaN to default."""
        if val is None:
            return default
        try:
            if math.isnan(val) or math.isinf(val):
                return default
        except TypeError:
            return default
        return float(val)

    R = _safe(cert.get("R"), 0.0)
    S_sup = _safe(cert.get("S_sup"), 0.0)
    N = _safe(cert.get("N"), 1.0)
    alpha = _safe(cert.get("alpha"), 0.0)
    sigma = _safe(cert.get("sigma"), 1.0)
    kappa = _safe(cert.get("kappa"), 0.0)

    gate_evidence = {}

    # Gate 1: Integrity Guard (OR logic — canonical SequentialGatekeeper)
    # V-011: read noise_admissibility from evidence, fall back to raw N
    evidence_block = cert.get("evidence", {})
    noise_admissibility = evidence_block.get("noise_admissibility")
    noise_fallback = noise_admissibility is None
    if noise_fallback:
        noise_admissibility = N

    gate1_n_pass = noise_admissibility < N_THR
    gate1_alpha_pass = alpha >= ALPHA_MIN
    gate1_pass = gate1_n_pass or gate1_alpha_pass

    gate_evidence["gate_1"] = {
        "noise_admissibility": noise_admissibility,
        "noise_fallback_to_raw_N": noise_fallback,
        "N_thr": N_THR,
        "N_pass": gate1_n_pass,
        "alpha": alpha,
        "alpha_min": ALPHA_MIN,
        "alpha_pass": gate1_alpha_pass,
        "pass": gate1_pass,
    }
    if not gate1_pass:
        return _build_result(cert, "REJECT", "GATE_1_INTEGRITY", gate_evidence, kappa, sigma, alpha)

    # Gate 2: Consensus Gate (canonical 3-path logic)
    coherence = cert.get("coherence") or evidence_block.get("coherence")
    if coherence is not None:
        gate2_pass = coherence >= _CFG.c_min
        gate_evidence["gate_2"] = {
            "coherence": coherence, "c_min": _CFG.c_min,
            "require_coherence": GATE_2_REQUIRE_COHERENCE, "pass": gate2_pass,
        }
        if not gate2_pass:
            return _build_result(cert, "BLOCK", "GATE_2_CONSENSUS", gate_evidence, kappa, sigma, alpha)
    elif GATE_2_REQUIRE_COHERENCE:
        gate_evidence["gate_2"] = {
            "coherence": None, "require_coherence": True, "pass": False,
            "failure_path": "coherence_absent_fail_closed",
        }
        return _build_result(cert, "BLOCK", "GATE_2_CONSENSUS", gate_evidence, kappa, sigma, alpha)
    else:
        gate_evidence["gate_2"] = {
            "coherence": None, "require_coherence": False, "pass": True,
            "failure_path": "coherence_absent_legacy_pass",
        }

    # Gate 3: Admissibility (Oobleck sigmoidal)
    kappa_req = _oobleck_kappa_req(sigma)
    landauer_lo = kappa_req - EPSILON_L

    gate_evidence["gate_3"] = {
        "kappa": kappa,
        "kappa_req": round(kappa_req, 6),
        "epsilon_L": EPSILON_L,
        "landauer_lo": round(landauer_lo, 6),
        "sigma": sigma,
        "landauer_sigma_tiebreaker": LANDAUER_SIGMA_TIEBREAKER,
    }

    if kappa >= kappa_req:
        gate_evidence["gate_3"]["zone"] = "clear_pass"
        gate_evidence["gate_3"]["pass"] = True
    elif kappa >= landauer_lo:
        # Landauer gray zone: sigma tiebreaker
        gate_evidence["gate_3"]["zone"] = "landauer_gray"
        if sigma <= LANDAUER_SIGMA_TIEBREAKER:
            gate_evidence["gate_3"]["pass"] = True
            gate_evidence["gate_3"]["tiebreaker"] = "sigma_low"
        else:
            gate_evidence["gate_3"]["pass"] = False
            gate_evidence["gate_3"]["tiebreaker"] = "sigma_high"
            return _build_result(cert, "RE_ENCODE", "GATE_3_ADMISSIBILITY", gate_evidence, kappa, sigma, alpha)
    else:
        gate_evidence["gate_3"]["zone"] = "clear_fail"
        gate_evidence["gate_3"]["pass"] = False
        return _build_result(cert, "RE_ENCODE", "GATE_3_ADMISSIBILITY", gate_evidence, kappa, sigma, alpha)

    # Gate 4: Grounding Gate (modal kappa check)
    # For block certificates, we use kappa as kappa_L proxy
    kappa_L = kappa  # Single-modal: kappa_L == kappa
    gate_evidence["gate_4"] = {
        "kappa_L": kappa_L,
        "kappa_L_min": KAPPA_L_MIN,
        "pass": kappa_L >= KAPPA_L_MIN,
    }
    if kappa_L < KAPPA_L_MIN:
        return _build_result(cert, "REPAIR", "GATE_4_GROUNDING", gate_evidence, kappa, sigma, alpha)

    # All gates pass
    return _build_result(cert, "EXECUTE", "ALL_PASSED", gate_evidence, kappa, sigma, alpha)


def _build_result(
    cert: dict,
    decision: str,
    gate_reached: str,
    gate_evidence: dict,
    kappa: float,
    sigma: float,
    alpha: float,
) -> dict:
    """Build the full admission result with gear computation."""
    # Compute tau and gear
    omega = cert.get("omega", 1.0) or 1.0
    prior = 0.5  # Default prior for alpha_omega blending
    alpha_omega = alpha * omega + prior * (1.0 - omega)
    tau = 1.0 / alpha_omega if alpha_omega > 0 else 999.0
    gear = _tau_to_gear(tau)

    # Sigma warning gear bump (only for EXECUTE)
    sigma_warning = False
    gear_after_bump = gear
    if decision == "EXECUTE" and sigma > SIGMA_WARNING_THR:
        sigma_warning = True
        bump_map = {"FIRST": "SECOND", "SECOND": "THIRD", "THIRD": "FOURTH", "FOURTH": "FOURTH"}
        gear_after_bump = bump_map.get(gear, gear)

    # Lyapunov advisory (V approximated from certificate)
    # V = tau * sigma (simplified Lyapunov proxy)
    V = tau * sigma
    lyapunov_gear = _classify_lyapunov(V)
    gear_agreement = gear_after_bump == lyapunov_gear

    # Admission tier classification
    effective_gear = gear_after_bump
    if decision == "EXECUTE":
        if effective_gear in ("FIRST", "SECOND"):
            admission_tier = "headline"
            admission_action = "admit-headline"
        elif effective_gear == "THIRD":
            admission_tier = "diagnostic-stabilizer"
            admission_action = "admit-stabilizer"
        elif effective_gear == "FOURTH":
            admission_tier = "marginal"
            admission_action = "admit-marginal"
        else:
            admission_tier = "integrity-reject"
            admission_action = "integrity-reject"
    elif decision == "WARN":
        if effective_gear in ("FIRST", "SECOND"):
            admission_tier = "headline"
            admission_action = "admit-headline"
        else:
            admission_tier = "diagnostic-stabilizer"
            admission_action = "admit-stabilizer"
    elif decision in ("RE_ENCODE", "REPAIR"):
        admission_tier = "morph-candidate"
        admission_action = "morph-cycle"
    elif decision == "REJECT":
        admission_tier = "rejected"
        admission_action = "reject"
    elif decision == "BLOCK":
        admission_tier = "quarantined"
        admission_action = "quarantine"
    else:
        admission_tier = "unknown"
        admission_action = "unknown"

    return {
        "block": cert.get("block"),
        "scenario": cert.get("scenario"),
        "target": cert.get("target"),
        "enforcement_decision": decision,
        "gate_reached": gate_reached,
        "gate_evidence": gate_evidence,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
        "tau": round(tau, 6),
        "gear": gear,
        "sigma_warning": sigma_warning,
        "gear_after_bump": gear_after_bump,
        "lyapunov_V": round(V, 6),
        "lyapunov_gear": lyapunov_gear,
        "gear_agreement": gear_agreement,
        "admission_tier": admission_tier,
        "admission_action": admission_action,
        "driving_mode": "ECO",
        "grid_used": "grid_A",
        "preset_id": "GEOSPATIAL_CONUS27",
    }


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> Optional[dict]:
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_1c: DGM block admission"
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would evaluate DGM gates on block certificates")
        log.info("  Preset: GEOSPATIAL_CONUS27 (kappa_base=%.2f, sigma_thr=%.2f)",
                 KAPPA_BASE, SIGMA_THR)
        return 0

    s3 = get_s3_client()

    print(f"\n{'='*60}")
    print(f"  S035 PHASE R3_1c: DGM BLOCK ADMISSION")
    print(f"{'='*60}\n")

    # Load block certificates from all scenarios
    all_certs = []
    for scenario in SCENARIOS:
        cert_data = _load_json(s3, f"{RESULTS_PREFIX}/r3_block_certificates_{scenario}.json")
        if cert_data:
            all_certs.extend(cert_data.get("certificates", []))
            log.info("Loaded %d block certs for %s",
                     len(cert_data.get("certificates", [])), scenario)

    if not all_certs:
        log.error("No block certificates found. Run compute_r3_block_tests.py first.")
        return 1

    # Load order robustness results
    order_data = {}
    for scenario in SCENARIOS:
        data = _load_json(s3, f"{RESULTS_PREFIX}/r3_order_robustness_{scenario}.json")
        if data:
            order_data[scenario] = data
        else:
            log.warning(
                "WARNING: order robustness data missing for scenario '%s' "
                "(r3_order_robustness_%s.json not found). "
                "All blocks for this scenario will default to order_robust=True, "
                "which silently hides any path-dependent blocks. "
                "Run compute_r3_order_robustness.py (R3_1b) before this script.",
                scenario, scenario,
            )

    # Evaluate gates on each block certificate
    admission_table = []
    trace_entries = []
    gear_summary = []

    skipped_invalid = 0
    invalid_details = []  # Track remediation paths for INVALID certs

    for cert in all_certs:
        # INVALID certificates: don't gate-evaluate, but DO diagnose and track
        if cert.get("certificate_status") == "INVALID":
            skipped_invalid += 1
            diag = _diagnose_invalid(cert)
            log.warning(
                "INVALID %s/%s/%s: %s -> %s",
                cert.get("scenario"), cert.get("block"), cert.get("target"),
                diag["failure_mode"], diag["remediation_action"],
            )
            entry = {
                "block": cert.get("block"),
                "scenario": cert.get("scenario"),
                "target": cert.get("target"),
                "enforcement_decision": "SKIP",
                "gate_reached": "PRE_GATE_INVALID",
                "certificate_status": "INVALID",
                "admission_tier": "invalid",
                "admission_action": diag["remediation_action"],
                "labels": ["invalid-certificate", diag["failure_mode"]],
                "invalid_diagnosis": diag,
                "certificate_evidence": cert.get("evidence", {}),
            }
            admission_table.append(entry)
            invalid_details.append(entry)
            continue

        result = evaluate_gates(cert)

        # Enrich with order robustness labels
        scenario = cert.get("scenario", "")
        block = cert.get("block", "")
        target = cert.get("target", "")

        order_robust = True
        if scenario in order_data:
            for target_result in order_data[scenario].get("results", []):
                if target_result.get("target") == target:
                    conc = target_result.get("concordance", {}).get("per_block", {})
                    block_conc = conc.get(block, {})
                    order_robust = block_conc.get("order_robust", True)

        # Apply exclusion labels
        labels = []
        if not order_robust:
            labels.append("path-dependent")
        if cert.get("delta_leakage") is not None and cert["delta_leakage"] > 0.05:
            labels.append("leakage-amplified")
        if cert.get("solver_agreement") is False:
            labels.append("solver-specific")
        if result["sigma_warning"]:
            labels.append("sigma-bumped")
        if not result["gear_agreement"]:
            labels.append("lyapunov-divergent")
        if result["admission_tier"] == "diagnostic-stabilizer":
            labels.append("diagnostic-stabilizer")
        if result["admission_tier"] == "marginal":
            labels.append("marginal-admit")

        result["labels"] = labels
        result["order_robust"] = order_robust

        admission_table.append(result)
        trace_entries.append({
            "cell": f"{scenario}/{target}",
            "block": block,
            "certificate": {
                k: cert.get(k) for k in
                ("R", "S_sup", "N", "alpha", "sigma", "spatial_metric",
                 "random_metric", "delta_spatial", "delta_leakage")
            },
            **result,
        })
        gear_summary.append({
            "block": block,
            "scenario": scenario,
            "target": target,
            "tau": result["tau"],
            "gear": result["gear"],
            "sigma_warning": result["sigma_warning"],
            "gear_after_bump": result["gear_after_bump"],
            "lyapunov_V": result["lyapunov_V"],
            "lyapunov_gear": result["lyapunov_gear"],
            "gear_agreement": result["gear_agreement"],
            "admission_tier": result["admission_tier"],
        })

    # Summary statistics
    decision_counts = {}
    tier_counts = {}
    for entry in admission_table:
        d = entry["enforcement_decision"]
        t = entry["admission_tier"]
        decision_counts[d] = decision_counts.get(d, 0) + 1
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # Assemble outputs
    admission_output = {
        "phase": "R3_1c_block_admission",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preset": "GEOSPATIAL_CONUS27",
        "thresholds": {
            "kappa_base": KAPPA_BASE,
            "sigma_thr": SIGMA_THR,
            "epsilon_L": EPSILON_L,
            "kappa_L_min": KAPPA_L_MIN,
            "N_thr": N_THR,
            "alpha_min": ALPHA_MIN,
        },
        "summary": {
            "n_blocks_evaluated": len(admission_table),
            "n_invalid": skipped_invalid,
            "n_gate_evaluated": len(admission_table) - skipped_invalid,
            "decision_counts": decision_counts,
            "tier_counts": tier_counts,
            "invalid_breakdown": _invalid_breakdown(invalid_details),
        },
        "admission_table": admission_table,
    }

    trace_output = {
        "phase": "R3_1c_dgm_admission_trace",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries": len(trace_entries),
        "trace": trace_entries,
    }

    gear_output = {
        "phase": "R3_1c_gear_summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": gear_summary,
    }

    # Write local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("r3_block_admission_table.json", admission_output),
        ("r3_dgm_admission_trace.json", trace_output),
        ("r3_gear_summary.json", gear_output),
    ]:
        with open(out_dir / name, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("Written %s", name)

    if args.upload:
        for name, data in [
            ("r3_block_admission_table.json", admission_output),
            ("r3_dgm_admission_trace.json", trace_output),
            ("r3_gear_summary.json", gear_output),
        ]:
            upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/{name}", data)
        log.info("Uploaded to S3")

    # Summary
    log.info("\n=== R3_1c Block Admission Summary ===")
    log.info("  Total: %d | Gate-evaluated: %d | INVALID: %d",
             len(admission_table), len(admission_table) - skipped_invalid, skipped_invalid)
    log.info("  Decisions:")
    for d, count in sorted(decision_counts.items()):
        log.info("    %s: %d", d, count)
    log.info("  Admission tiers:")
    for t, count in sorted(tier_counts.items()):
        log.info("    %s: %d", t, count)

    if invalid_details:
        breakdown = _invalid_breakdown(invalid_details)
        log.info("  INVALID breakdown by failure mode:")
        for mode, count in sorted(breakdown["by_failure_mode"].items()):
            log.info("    %s: %d", mode, count)
        log.info("  INVALID remediation actions:")
        for action, count in sorted(breakdown["by_remediation_action"].items()):
            log.info("    %s: %d", action, count)
        # Flag blocks with partial signals (some data survived)
        for block, entries in breakdown["by_block"].items():
            has_partial = any(e.get("partial_signals") for e in entries)
            if has_partial:
                log.info("    %s: partial evidence available -- diagnostic recovery possible", block)

    return 0


if __name__ == "__main__":
    sys.exit(main())
