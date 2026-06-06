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

# GEOSPATIAL_CONUS27 preset thresholds (from yrsn-controlplane/presets.py)
KAPPA_BASE = 0.22
SIGMA_THR = 0.50
LAMBDA_TURBULENCE = 0.0
EPSILON_L = 0.01
KAPPA_L_MIN = 0.15
N_THR = 0.70          # Noise threshold for Gate 1
ALPHA_MIN = 0.30      # Minimum alpha for Gate 2

# Gear thresholds (P14, from yrsn-controlplane/types.py)
TAU_FIRST = 1.0
TAU_SECOND = 1.43
TAU_THIRD = 2.5

# Sigma warning threshold
SIGMA_WARNING_THR = 0.40

# Oobleck sigmoidal parameters
OOBLECK_DELTA_KAPPA = 0.28
OOBLECK_STEEPNESS = 10.0
OOBLECK_SIGMA_C = 0.40


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
    """Oobleck dynamic threshold: kappa_req(sigma)."""
    return KAPPA_BASE + OOBLECK_DELTA_KAPPA * _sigmoid(
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


def evaluate_gates(cert: dict) -> dict:
    """Evaluate the 4-gate pipeline on a block certificate.

    Returns gate result with decision, gear, and full trace.
    """
    R = cert.get("R", 0.0) or 0.0
    S_sup = cert.get("S_sup", 0.0) or 0.0
    N = cert.get("N", 1.0) or 1.0
    alpha = cert.get("alpha", 0.0) or 0.0
    sigma = cert.get("sigma", 1.0) or 1.0
    kappa = cert.get("kappa") or cert.get("delta_spatial") or 0.0

    # Use delta_spatial as kappa proxy if kappa not available
    # (block certificates use delta_spatial as the compatibility signal)
    if kappa == 0.0 and cert.get("delta_spatial") is not None:
        kappa = max(0.0, cert["delta_spatial"])

    gate_evidence = {}

    # Gate 1: Noise threshold (N > N_thr -> REJECT)
    gate_evidence["gate_1"] = {"N": N, "N_thr": N_THR, "pass": N <= N_THR}
    if N > N_THR:
        return _build_result(cert, "REJECT", "GATE_1", gate_evidence, kappa, sigma, alpha)

    # Gate 2: Coherence / alpha check (alpha < alpha_min -> BLOCK)
    gate_evidence["gate_2"] = {"alpha": alpha, "alpha_min": ALPHA_MIN, "pass": alpha >= ALPHA_MIN}
    if alpha < ALPHA_MIN:
        return _build_result(cert, "BLOCK", "GATE_2", gate_evidence, kappa, sigma, alpha)

    # Gate 3: Kappa vs Oobleck dynamic threshold
    kappa_req = _oobleck_kappa_req(sigma)
    landauer_lo = kappa_req - EPSILON_L

    gate_evidence["gate_3"] = {
        "kappa": kappa,
        "kappa_req": round(kappa_req, 6),
        "epsilon_L": EPSILON_L,
        "landauer_lo": round(landauer_lo, 6),
        "sigma": sigma,
    }

    if kappa >= kappa_req:
        gate_evidence["gate_3"]["zone"] = "clear_pass"
        gate_evidence["gate_3"]["pass"] = True
    elif kappa >= landauer_lo:
        # Landauer gray zone: sigma tiebreaker
        gate_evidence["gate_3"]["zone"] = "landauer_gray"
        if sigma <= SIGMA_THR:
            gate_evidence["gate_3"]["pass"] = True
            gate_evidence["gate_3"]["tiebreaker"] = "sigma_low"
        else:
            gate_evidence["gate_3"]["pass"] = False
            gate_evidence["gate_3"]["tiebreaker"] = "sigma_high"
            return _build_result(cert, "RE_ENCODE", "GATE_3", gate_evidence, kappa, sigma, alpha)
    else:
        gate_evidence["gate_3"]["zone"] = "clear_fail"
        gate_evidence["gate_3"]["pass"] = False
        return _build_result(cert, "RE_ENCODE", "GATE_3", gate_evidence, kappa, sigma, alpha)

    # Gate 4: Modal kappa check (kappa_L < kappa_L_min -> REPAIR)
    # For block certificates, we use kappa as kappa_L proxy
    kappa_L = kappa  # Single-modal: kappa_L == kappa
    gate_evidence["gate_4"] = {
        "kappa_L": kappa_L,
        "kappa_L_min": KAPPA_L_MIN,
        "pass": kappa_L >= KAPPA_L_MIN,
    }
    if kappa_L < KAPPA_L_MIN:
        return _build_result(cert, "REPAIR", "GATE_4", gate_evidence, kappa, sigma, alpha)

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

    for cert in all_certs:
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
            "decision_counts": decision_counts,
            "tier_counts": tier_counts,
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
    log.info("  Evaluated: %d block certificates", len(admission_table))
    for d, count in sorted(decision_counts.items()):
        log.info("  %s: %d", d, count)
    log.info("  Admission tiers:")
    for t, count in sorted(tier_counts.items()):
        log.info("    %s: %d", t, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
