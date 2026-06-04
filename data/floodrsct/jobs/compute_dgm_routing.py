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
(scenario, target) cell using RSCT certificates at R0/R1/R2.  Compares
the recommended arm against the actual best arm.  Tests H5: can
certificate-driven routing select the right representation level?

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

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, level_prefix
from _s3_result import upload_json_result

from yrsn.core.dgm_unified import MorphType

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
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    return index


def _load_results(s3, level: str, scenario: str) -> dict:
    """Load model results, indexed by target."""
    data = _load_json(s3, f"{RESULTS_PREFIX}/{level_prefix(level)}_{scenario}.json")
    if not data:
        return {}
    return {cell["target"]: cell for cell in data.get("cells", [])}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def apply_routing(certs: dict[str, Optional[dict]]) -> tuple[Optional[str], str]:
    """Apply DGM routing decision tree.

    Returns (recommended_arm, morph_decision).
    recommended_arm is None for REPAIR/PRUNING (no single arm recommended).
    """
    c0 = certs.get("r0")
    if c0 and c0.get("kappa", 0) >= KAPPA_GOOD:
        return "r0", MorphType.VERIFICATION.value

    if c0 and c0.get("S_sup", 0) > S_SUP_HIGH:
        c1 = certs.get("r1")
        if c1 and c1.get("kappa", 0) >= KAPPA_GOOD:
            return "r1", MorphType.RE_ENCODE.value if hasattr(MorphType, "RE_ENCODE") else "re_encode"

    c1 = certs.get("r1")
    diag_transfer = c1.get("diag_transfer", 1.0) if c1 else 1.0
    if diag_transfer < DIAG_TRANSFER_LOW:
        c2 = certs.get("r2")
        if c2 and c2.get("kappa", 0) >= KAPPA_GOOD:
            return "r2", MorphType.RE_ENCODE.value if hasattr(MorphType, "RE_ENCODE") else "re_encode"

    # Fallback: check R2 cert for terminal decisions
    c2 = certs.get("r2")
    if c2:
        if c2.get("sigma", 0) > SIGMA_HIGH:
            return None, MorphType.REPAIR.value
        if c2.get("kappa", 0) < KAPPA_BAD:
            return None, MorphType.PRUNING.value
        # R2 kappa is between BAD and GOOD -- recommend R2 as best available
        return "r2", MorphType.ENSEMBLE.value

    # No R2 cert available -- fall back to best available
    if c1:
        return "r1", "fallback"
    return "r0", "fallback"


def determine_best_arm(results: dict[str, Optional[dict]], target: str) -> tuple[Optional[str], Optional[float]]:
    """Find the level with the best primary metric for a target."""
    metrics = {}
    for level in LEVELS:
        cell = results.get(level, {}).get(target)
        if cell is None:
            continue
        # Binary classification uses roc_auc, regression uses R2/spatial_metric
        if "spatial_roc_auc" in cell:
            metrics[level] = cell["spatial_roc_auc"]
        elif "spatial_metric" in cell:
            metrics[level] = cell["spatial_metric"]

    if not metrics:
        return None, None

    best = max(metrics, key=metrics.get)
    return best, metrics[best]


def binomial_ci(hits: int, n: int) -> tuple[float, float]:
    """95% binomial confidence interval for hit rate."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    lo, hi = binom.interval(0.95, n, p)
    return float(lo / n), float(hi / n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6: DGM routing proof-of-concept")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip execution")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would load certificates + results for R0/R1/R2")
        log.info("Apply routing decision tree per cell, compare vs actual best arm")
        log.info("Writes: %s/dgm_routing.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()

    # Load all certificates (cached per level)
    cert_index = {}
    for level in LEVELS:
        cert_index[level] = _load_certificates(s3, level)
        log.info("Loaded %d certificates for %s", len(cert_index[level]), level)

    # Discover cells from R0 certificates
    cells = list(cert_index["r0"].keys())
    if not cells:
        log.error("No cells found in R0 certificates")
        return 1
    log.info("Processing %d cells", len(cells))

    # Load all results (cached per level+scenario)
    result_index: dict[str, dict] = {}
    scenarios = sorted({s for s, _ in cells})
    for level in LEVELS:
        result_index[level] = {}
        for scenario in scenarios:
            result_index[level].update(
                {scenario: _load_results(s3, level, scenario)}
            )

    # Route each cell
    routing_table = []
    correct_count = 0
    near_optimal_count = 0

    for scenario, target in cells:
        certs = {lv: cert_index[lv].get((scenario, target)) for lv in LEVELS}

        if all(v is None for v in certs.values()):
            log.warning("No certificates for %s/%s -- skipping", scenario, target)
            continue

        recommended_arm, morph_decision = apply_routing(certs)

        # Actual best arm from results
        level_results = {lv: result_index.get(lv, {}).get(scenario, {}) for lv in LEVELS}
        actual_best, best_metric = determine_best_arm(level_results, target)

        # Correctness
        correct = recommended_arm is not None and recommended_arm == actual_best

        # Near-optimal check
        near_optimal = False
        if recommended_arm and actual_best and best_metric is not None:
            rec_cell = level_results.get(recommended_arm, {}).get(target, {})
            rec_metric = rec_cell.get("spatial_roc_auc", rec_cell.get("spatial_metric"))
            if rec_metric is not None:
                near_optimal = abs(best_metric - rec_metric) <= NEAR_OPTIMAL_DELTA

        if correct:
            correct_count += 1
        if near_optimal or correct:
            near_optimal_count += 1

        row = {
            "scenario": scenario,
            "target": target,
            "cert_r0": {k: certs["r0"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r0"] else None,
            "cert_r1": {k: certs["r1"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r1"] else None,
            "cert_r2": {k: certs["r2"].get(k) for k in ("kappa", "S_sup", "sigma", "R", "N")} if certs["r2"] else None,
            "morph_decision": morph_decision,
            "recommended_arm": recommended_arm,
            "actual_best_arm": actual_best,
            "correct": correct,
            "near_optimal": near_optimal or correct,
        }
        routing_table.append(row)
        log.info(
            "  %s/%s: recommend=%s actual=%s %s",
            scenario, target,
            recommended_arm or morph_decision,
            actual_best or "?",
            "HIT" if correct else ("NEAR" if near_optimal else "MISS"),
        )

    n_cells = len(routing_table)
    hit_rate = correct_count / n_cells if n_cells else 0.0
    ci_lo, ci_hi = binomial_ci(correct_count, n_cells)

    result = {
        "phase": "6_dgm_routing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells": n_cells,
        "routing_table": routing_table,
        "summary": {
            "hit_rate": round(hit_rate, 4),
            "hit_rate_ci_lower": round(ci_lo, 4),
            "hit_rate_ci_upper": round(ci_hi, 4),
            "near_optimal_rate": round(near_optimal_count / n_cells, 4) if n_cells else 0.0,
            "n_cells": n_cells,
            "caveat": "n=7 cells; proof-of-concept only, not powered for inference",
        },
        "thresholds": {
            "kappa_good": KAPPA_GOOD,
            "s_sup_high": S_SUP_HIGH,
            "diag_transfer_low": DIAG_TRANSFER_LOW,
            "sigma_high": SIGMA_HIGH,
            "kappa_bad": KAPPA_BAD,
            "near_optimal_delta": NEAR_OPTIMAL_DELTA,
        },
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
    log.info("\n=== DGM Routing Summary ===")
    log.info("  Hit rate: %d/%d = %.1f%% [%.1f%%, %.1f%%]",
             correct_count, n_cells, hit_rate * 100, ci_lo * 100, ci_hi * 100)
    log.info("  Near-optimal: %d/%d = %.1f%%",
             near_optimal_count, n_cells,
             near_optimal_count / n_cells * 100 if n_cells else 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
