#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: kimi (Moonshot AI, via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Fix upload_json_result signature, split process_all_cells,
#       cache certificate loads, fix REPAIR/PRUNING arm handling, add shebang/
#       docstring, align logging/timestamp with template, add dry-run gate
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""compute_dgm_routing.py -- Phase 6: DGM routing proof-of-concept.

Applies the DGM (Dual Graph Model) routing decision tree to each
(scenario, target) cell using RSCT certificates at R0/R1/R2 plus
gearbox warmup signals (Phase 0.75).  Compares the recommended arm
against the actual best arm.  Tests H5: can certificate-driven routing
select the right representation level?

Two routing strategies are evaluated:
  1. kappa-first (legacy): uses kappa_geom as primary discriminant.
     Known limitation: kappa_geom is level-invariant, so it always
     exits on the first branch (R0/VERIFICATION) when kappa >= 0.7.
  2. gear-informed (new): uses gearbox warmup signals (gear, alpha,
     sigma) which vary per cell, enabling actual discrimination.

Framed as proof-of-concept (n=7 cells is too small for inference).

Usage:
    python compute_dgm_routing.py --upload
    python compute_dgm_routing.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from georsct.domain.certificate import InternalDecision, PublicDecision
except ImportError:
    # SageMaker: georsct package not installed; define enums inline
    from enum import Enum

    class PublicDecision(str, Enum):
        EXECUTE = "EXECUTE"
        CAUTION = "CAUTION"
        REFUSE = "REFUSE"

    class InternalDecision(str, Enum):
        EXECUTE = "EXECUTE"
        REJECT = "REJECT"
        BLOCK = "BLOCK"
        RE_ENCODE = "RE_ENCODE"
        REPAIR = "REPAIR"
        WARN = "WARN"
        FALLBACK = "FALLBACK"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
LEVELS = ["r0", "r1", "r2"]

# Routing thresholds (from DOE_certificate_dgm.md)
KAPPA_GOOD = 0.7
S_SUP_HIGH = 0.2
DIAG_TRANSFER_LOW = 0.5
SIGMA_HIGH = 0.15
KAPPA_BAD = 0.3
NEAR_OPTIMAL_DELTA = 0.02

# Gear-informed routing thresholds.
# Derived from the RSCT quality gate system (DOE_certificate_dgm.md):
#   ALPHA_WARMUP_HIGH = 0.3: matches Gate 1 R >= 0.3 (integrity floor).
#     If a cheap Ridge solver exceeds the integrity floor, the cell
#     is "easy enough" that R0 representation likely suffices.
#   ALPHA_WARMUP_LOW = 0.1: below this, even the cheap solver struggles;
#     the cell needs more representation power (R1/R2).
#     Derived from: R < 0.3 is Gate 1 fail; 0.1 is the "clearly below
#     chance" zone for regression R2 or rescaled AUC.
#   SIGMA_STABLE = 0.10: fold sigma below this indicates stable CV
#     performance. Derived from: sigma_high (0.15) is the REPAIR
#     threshold; 0.10 is 2/3 of that, used as the "clearly stable" zone.
# These thresholds will be replaced by data-calibrated values from
# gearbox_warmup.json when the oobleck autocalibration is wired in.
ALPHA_WARMUP_HIGH = 0.3
ALPHA_WARMUP_LOW = 0.1
SIGMA_STABLE = 0.10


def _v(d: Optional[dict], key: str, default: float) -> float:
    """Get value from dict, coalescing None to default."""
    if d is None:
        return default
    val = d.get(key)
    return val if val is not None else default


# InternalDecision -> PublicDecision projection (ADR-029/034).
# System-internal routing decisions map to user-facing operational guidance.
DECISION_PROJECTION = {
    InternalDecision.EXECUTE:   PublicDecision.EXECUTE,
    InternalDecision.REJECT:    PublicDecision.REFUSE,
    InternalDecision.BLOCK:     PublicDecision.REFUSE,
    InternalDecision.RE_ENCODE: PublicDecision.CAUTION,
    InternalDecision.REPAIR:    PublicDecision.REFUSE,
    InternalDecision.WARN:      PublicDecision.CAUTION,
    InternalDecision.FALLBACK:  PublicDecision.CAUTION,
}


def project_public_decision(internal: InternalDecision) -> PublicDecision:
    """Project an internal gatekeeper decision to a public decision."""
    return DECISION_PROJECTION.get(internal, PublicDecision.REFUSE)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> Optional[dict]:
    """Load JSON from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_certificates(s3, level: str) -> dict:
    """Load certificates for a level, indexed by (scenario, target)."""
    data = _load_json(s3, f"{RESULTS_PREFIX}/certificates_{level}.json")
    if not data:
        return {}
    index = {}
    # Certificates JSON uses "certificates" key, not "cells"
    certs = data.get("certificates", data.get("cells", []))
    for cell in certs:
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    return index


def _load_gearbox_warmup(s3) -> dict:
    """Load Phase 0.75 gearbox warmup, indexed by (scenario, target)."""
    data = _load_json(s3, f"{RESULTS_PREFIX}/gearbox_warmup.json")
    if not data:
        log.warning(
            "gearbox_warmup.json not found -- gear-informed routing unavailable. "
            "Run compute_gearbox_warmup.py first."
        )
        return {}
    index = {}
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    log.info("Loaded gearbox warmup for %d cells", len(index))
    return index


# ---------------------------------------------------------------------------
# Routing logic: kappa-first (legacy)
# ---------------------------------------------------------------------------

def apply_routing_kappa(certs: dict[str, Optional[dict]]) -> tuple[Optional[str], str]:
    """Apply legacy kappa-first DGM routing decision tree.

    Known limitation: kappa_geom is level-invariant, so this always
    exits on the first branch when kappa >= 0.7 (all cells).

    Returns (recommended_arm, internal_decision).
    recommended_arm is None for REPAIR/REJECT (no single arm recommended).
    """
    c0 = certs.get("r0")
    if c0 and _v(c0, "kappa", 0) >= KAPPA_GOOD:
        return "r0", InternalDecision.EXECUTE.value

    if c0 and _v(c0, "S_sup", 0) > S_SUP_HIGH:
        c1 = certs.get("r1")
        if c1 and _v(c1, "kappa", 0) >= KAPPA_GOOD:
            return "r1", InternalDecision.RE_ENCODE.value

    c1 = certs.get("r1")
    diag_transfer = _v(c1, "diag_transfer", 1.0)
    if diag_transfer < DIAG_TRANSFER_LOW:
        c2 = certs.get("r2")
        if c2 and _v(c2, "kappa", 0) >= KAPPA_GOOD:
            return "r2", InternalDecision.RE_ENCODE.value

    # Fallback: check R2 cert for terminal decisions
    c2 = certs.get("r2")
    if c2:
        if _v(c2, "sigma", 0) > SIGMA_HIGH:
            return None, InternalDecision.REPAIR.value
        if _v(c2, "kappa", 0) < KAPPA_BAD:
            return None, InternalDecision.REJECT.value
        # R2 kappa is between BAD and GOOD -- recommend R2 as best available
        return "r2", InternalDecision.WARN.value

    # No R2 cert available -- fall back to best available
    if c1:
        return "r1", InternalDecision.FALLBACK.value
    return "r0", InternalDecision.FALLBACK.value


# ---------------------------------------------------------------------------
# Routing logic: gear-informed (new)
# ---------------------------------------------------------------------------

def apply_routing_gear(
    certs: dict[str, Optional[dict]],
    warmup: Optional[dict],
) -> tuple[Optional[str], str]:
    """Apply gear-informed DGM routing decision tree.

    Uses gearbox warmup signals (gear, alpha_warmup, sigma, collapse_risk)
    as primary discriminant. These vary per cell, unlike kappa_geom.

    Decision logic:
      1. REVERSE gear (collapse_risk > 0.8) -> REPAIR
      2. FIRST gear + high alpha_warmup + low sigma -> R0 / VERIFICATION
         (easy cell, cheap solver suffices)
      3. THIRD/FOURTH gear or low alpha_warmup -> compare R1/R2 certificates
         (hard cell, needs more representation power)
      4. SECOND gear -> compare R0 vs R1 certificates (borderline)

    Returns (recommended_arm, internal_decision).
    """
    # No warmup data -> fall back to certificate-only comparison
    if warmup is None:
        return _route_by_certificate_comparison(certs)

    gear = warmup.get("gear") if warmup.get("gear") is not None else 2
    alpha_w = warmup.get("alpha_warmup") if warmup.get("alpha_warmup") is not None else 0.0
    sigma_w = warmup.get("sigma") if warmup.get("sigma") is not None else 1.0
    collapse = warmup.get("collapse_risk") if warmup.get("collapse_risk") is not None else 0.0

    # 1. Collapse -> REPAIR (no arm can help)
    if gear == -1 or collapse > 0.8:
        return None, InternalDecision.REPAIR.value

    # 2. FIRST gear: easy cell, R0 likely sufficient
    if gear == 1 and alpha_w >= ALPHA_WARMUP_HIGH and sigma_w <= SIGMA_STABLE:
        c0 = certs.get("r0")
        if c0 and _v(c0, "R", 0) > 0.3:
            return "r0", InternalDecision.EXECUTE.value
        # R0 cert doesn't confirm -- escalate to comparison
        return _route_by_certificate_comparison(certs)

    # 3. THIRD/FOURTH gear or very low alpha: hard cell, needs representation power
    if gear >= 3 or alpha_w < ALPHA_WARMUP_LOW:
        # Use full simplex: R dominates AND N is contained (R > S_sup AND N < 0.5).
        # Ignoring N would treat a high-R, high-N cell as adequate when the
        # noise floor actually invalidates it.
        c2 = certs.get("r2")
        if c2 and _v(c2, "R", 0) > _v(c2, "S_sup", 1.0) and _v(c2, "N", 1.0) < 0.5:
            return "r2", InternalDecision.RE_ENCODE.value
        c1 = certs.get("r1")
        if c1 and _v(c1, "R", 0) > _v(c1, "S_sup", 1.0) and _v(c1, "N", 1.0) < 0.5:
            return "r1", InternalDecision.RE_ENCODE.value
        # No arm shows R-dominant with contained N -- pruning candidate
        if c2 and _v(c2, "kappa", 0) < KAPPA_BAD:
            return None, InternalDecision.REJECT.value
        return _route_by_certificate_comparison(certs)

    # 4. SECOND gear: borderline -- compare certificates, but respect
    # warmup signal: if warmup alpha is moderate, prefer R0 when the
    # certificate comparison would select R0 anyway (no contradiction).
    # If certificate comparison selects R1/R2, the warmup "borderline"
    # signal doesn't override -- certificates are post-training evidence.
    return _route_by_certificate_comparison(certs)


def _route_by_certificate_comparison(
    certs: dict[str, Optional[dict]],
) -> tuple[Optional[str], str]:
    """Route by comparing spatial_metric across arms.

    Selects the arm with the best spatial_metric. Falls back to R0
    if no certificates are available.
    """
    best_arm, best_metric = determine_best_arm(certs)
    if best_arm is None:
        return "r0", InternalDecision.FALLBACK.value

    # Best arm from certificate comparison
    if best_arm == "r0":
        return "r0", InternalDecision.EXECUTE.value
    elif best_arm in ("r1", "r2"):
        return best_arm, InternalDecision.RE_ENCODE.value
    return best_arm, InternalDecision.FALLBACK.value


def determine_best_arm(certs: dict[str, Optional[dict]]) -> tuple[Optional[str], Optional[float]]:
    """Find the level with the best spatial_metric from certificates."""
    metrics = {}
    for level in LEVELS:
        cert = certs.get(level)
        if cert is None:
            continue
        val = cert.get("spatial_metric")
        if val is not None:
            metrics[level] = val

    if not metrics:
        return None, None

    best = max(metrics, key=metrics.get)
    return best, metrics[best]


def binomial_ci(hits: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson exact 95% confidence interval for hit rate.

    binom.interval(alpha, n, p) computes a prediction interval for
    future counts given a known p -- NOT a confidence interval for
    an unknown proportion. For small n (n~7 in this PoC), the
    distinction matters. Clopper-Pearson inverts the binomial test
    to get a proper CI on the true proportion.
    """
    if n == 0:
        return 0.0, 0.0
    alpha = 0.05
    lo = beta_dist.ppf(alpha / 2, hits, n - hits + 1) if hits > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, hits + 1, n - hits) if hits < n else 1.0
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _evaluate_strategy(
    strategy_name: str,
    route_fn,
    cells: list[tuple[str, str]],
    cert_index: dict,
    warmup_index: dict,
) -> tuple[list[dict], dict]:
    """Evaluate one routing strategy across all cells.

    Returns (routing_table, summary_dict).
    """
    routing_table = []
    correct_count = 0
    near_optimal_count = 0

    for scenario, target in cells:
        certs = {lv: cert_index[lv].get((scenario, target)) for lv in LEVELS}

        if all(v is None for v in certs.values()):
            log.warning("No certificates for %s/%s -- skipping", scenario, target)
            continue

        # Route using the specified strategy
        if strategy_name == "gear_informed":
            warmup = warmup_index.get((scenario, target))
            recommended_arm, internal_decision = route_fn(certs, warmup)
        else:
            recommended_arm, internal_decision = route_fn(certs)
        morph_decision = internal_decision  # legacy alias for output key

        # Actual best arm from certificate spatial_metric
        actual_best, best_metric = determine_best_arm(certs)

        # Correctness
        correct = recommended_arm is not None and recommended_arm == actual_best

        # Near-optimal check
        near_optimal = False
        if recommended_arm and actual_best and best_metric is not None:
            rec_cert = certs.get(recommended_arm)
            rec_metric = rec_cert.get("spatial_metric") if rec_cert else None
            if rec_metric is not None:
                near_optimal = abs(best_metric - rec_metric) <= NEAR_OPTIMAL_DELTA

        if correct:
            correct_count += 1
        if near_optimal or correct:
            near_optimal_count += 1

        # Project internal decision to public decision (ADR-029/034)
        try:
            internal = InternalDecision(morph_decision)
        except ValueError:
            internal = InternalDecision.FALLBACK
        public = project_public_decision(internal)

        row = {
            "scenario": scenario,
            "target": target,
            "cert_r0": {k: certs["r0"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r0"] else None,
            "cert_r1": {k: certs["r1"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r1"] else None,
            "cert_r2": {k: certs["r2"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r2"] else None,
            "internal_decision": morph_decision,
            "public_decision": public.value,
            "recommended_arm": recommended_arm,
            "actual_best_arm": actual_best,
            "correct": correct,
            "near_optimal": near_optimal or correct,
        }

        # Add warmup signals if available
        if strategy_name == "gear_informed":
            warmup = warmup_index.get((scenario, target))
            if warmup:
                row["warmup_gear"] = warmup.get("gear_name")
                row["warmup_alpha"] = warmup.get("alpha_warmup")
                row["warmup_sigma"] = warmup.get("sigma")

        routing_table.append(row)
        log.info(
            "  [%s] %s/%s: recommend=%s actual=%s %s",
            strategy_name, scenario, target,
            recommended_arm or morph_decision,
            actual_best or "?",
            "HIT" if correct else ("NEAR" if near_optimal else "MISS"),
        )

    n_cells = len(routing_table)
    hit_rate = correct_count / n_cells if n_cells else 0.0
    ci_lo, ci_hi = binomial_ci(correct_count, n_cells)

    summary = {
        "strategy": strategy_name,
        "hit_rate": round(hit_rate, 4),
        "hit_rate_ci_lower": round(ci_lo, 4),
        "hit_rate_ci_upper": round(ci_hi, 4),
        "near_optimal_rate": round(near_optimal_count / n_cells, 4) if n_cells else 0.0,
        "correct_count": correct_count,
        "near_optimal_count": near_optimal_count,
        "n_cells": n_cells,
        "caveat": "proof-of-concept; not powered for inference",
    }

    return routing_table, summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6: DGM routing proof-of-concept")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip execution")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would load certificates + gearbox warmup for R0/R1/R2")
        log.info("Apply two routing strategies (kappa-first, gear-informed) per cell")
        log.info("Writes: %s/dgm_routing.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()

    # Load all certificates (cached per level)
    cert_index = {}
    for level in LEVELS:
        cert_index[level] = _load_certificates(s3, level)
        log.info("Loaded %d certificates for %s", len(cert_index[level]), level)

    # Load gearbox warmup (Phase 0.75)
    warmup_index = _load_gearbox_warmup(s3)

    # Discover cells from R0 certificates
    cells = list(cert_index["r0"].keys())
    if not cells:
        log.error("No cells found in R0 certificates")
        return 1
    log.info("Processing %d cells", len(cells))

    # --- Strategy 1: kappa-first (legacy) ---
    log.info("\n=== Strategy: kappa-first (legacy) ===")
    kappa_table, kappa_summary = _evaluate_strategy(
        "kappa_first", apply_routing_kappa, cells, cert_index, warmup_index,
    )

    # --- Strategy 2: gear-informed (new) ---
    log.info("\n=== Strategy: gear-informed ===")
    gear_table, gear_summary = _evaluate_strategy(
        "gear_informed", apply_routing_gear, cells, cert_index, warmup_index,
    )

    result = {
        "phase": "6_dgm_routing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells),
        "strategies": {
            "kappa_first": {
                "routing_table": kappa_table,
                "summary": kappa_summary,
            },
            "gear_informed": {
                "routing_table": gear_table,
                "summary": gear_summary,
            },
        },
        "comparison": {
            "kappa_first_hit_rate": kappa_summary["hit_rate"],
            "gear_informed_hit_rate": gear_summary["hit_rate"],
            "improvement": round(gear_summary["hit_rate"] - kappa_summary["hit_rate"], 4),
        },
        "thresholds": {
            "kappa_good": KAPPA_GOOD,
            "s_sup_high": S_SUP_HIGH,
            "diag_transfer_low": DIAG_TRANSFER_LOW,
            "sigma_high": SIGMA_HIGH,
            "kappa_bad": KAPPA_BAD,
            "near_optimal_delta": NEAR_OPTIMAL_DELTA,
            "alpha_warmup_high": ALPHA_WARMUP_HIGH,
            "alpha_warmup_low": ALPHA_WARMUP_LOW,
            "sigma_stable": SIGMA_STABLE,
        },
        "gearbox_warmup_available": bool(warmup_index),
    }

    # Write local copy
    out_path = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_path.mkdir(parents=True, exist_ok=True)
    local_file = out_path / "dgm_routing.json"
    with open(local_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Written to %s", local_file)

    if args.upload:
        key = f"{RESULTS_PREFIX}/dgm_routing.json"
        upload_json_result(s3, BUCKET, key, result)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Summary
    log.info("\n=== DGM Routing Comparison ===")
    for name, s in [("kappa-first", kappa_summary), ("gear-informed", gear_summary)]:
        log.info("  %s: %d/%d = %.1f%% [%.1f%%, %.1f%%] | near-optimal: %d/%d = %.1f%%",
                 name, s["correct_count"], s["n_cells"],
                 s["hit_rate"] * 100, s["hit_rate_ci_lower"] * 100, s["hit_rate_ci_upper"] * 100,
                 s["near_optimal_count"], s["n_cells"], s["near_optimal_rate"] * 100)
    imp = result["comparison"]["improvement"]
    log.info("  Improvement: %+.1f pp", imp * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
