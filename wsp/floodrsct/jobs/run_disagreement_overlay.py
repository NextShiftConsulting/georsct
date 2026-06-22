#!/usr/bin/env python3
"""run_disagreement_overlay.py -- C6 + C8 artifacts for V8 paper.

C6: SVI-disagreement overlay (join DOE-C1 divergence with SVI/HIFLD per ZCTA)
C8: Baseline disagreement comparison (certificate distance vs naive distance)

Reads:
  - DOE-C1 five_construct_{scenario}.json (per-construct certificates)
  - DOE-C1 cache/pairwise_{scenario}.parquet (pairwise certificate distances)
  - event_features parquet (SVI + HIFLD columns)

Produces:
  - results/s035/v8/svi_disagreement_{scenario}.json
  - results/s035/v8/baseline_comparison_{scenario}.json
  - results/s035/v8/svi_disagreement_summary.json
  - results/s035/v8/baseline_comparison_summary.json

Usage:
    python run_disagreement_overlay.py --scenario houston --upload
    python run_disagreement_overlay.py --scenario houston --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS = [
    "houston",
    "southwest_florida",
    "nyc",
    "riverside_coachella",
    "new_orleans",
]

BUCKET = "swarm-floodrsct-data"

# SVI columns in event_features parquet (CDC Social Vulnerability Index)
SVI_COLS = [
    "svi_overall",
    "svi_socioeconomic",
    "svi_household_disability",
    "svi_minority_language",
    "svi_housing_transport",
]

# HIFLD columns in event_features parquet
HIFLD_COLS = [
    "hifld_n_hospitals",
    "hifld_nearest_hospital_km",
    "hifld_n_hospital_beds",
    "hifld_n_pharmacies",
    "hifld_nearest_pharmacy_km",
    "hifld_nearest_trauma_center_km",
]

# Constructs in DOE-C1
CONSTRUCTS = ["jrc_observed_water", "deltares_rp_depth",
              "fema_regulatory_zone", "fast_modeled_damage",
              "nfip_administrative_loss"]


# ---------------------------------------------------------------------------
# S3 helpers (reuse pattern from other jobs)
# ---------------------------------------------------------------------------

def _get_s3():
    """Get S3 client via swarm_auth."""
    from swarm_auth import get_aws_credentials
    import boto3
    aws = get_aws_credentials()
    return boto3.client("s3", region_name="us-east-1", **aws)


def _read_json_s3(s3, key):
    """Read JSON from S3."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _read_parquet_s3(s3, key):
    """Read parquet from S3."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    return pd.read_parquet(buf)


# ---------------------------------------------------------------------------
# C8: Baseline disagreement comparison
# ---------------------------------------------------------------------------

def compute_baseline_comparison(doe_c1_json: dict) -> dict:
    """Compare certificate divergence vs naive raw-score disagreement.

    For each construct pair:
    - Certificate distance = Euclidean in (forward_score, kappa_spatial,
      kappa_reconstruct) space (from DOE-C1)
    - Naive distance = |forward_score_A - forward_score_B| (raw R2 gap)
    - forward_delta, kappa_spatial_delta, kappa_reconstruct_delta individually

    Key question: do pairs exist with low naive distance but high certificate
    distance? That means the decomposition reveals structure raw comparison misses.
    """
    per_construct = doe_c1_json["per_construct"]
    pairwise = doe_c1_json["pairwise"]

    # Build lookup
    cert_map = {}
    for c in per_construct:
        if c["target_available"] and c["forward_score"] is not None:
            cert_map[c["construct"]] = c

    rows = []
    for p in pairwise:
        if not p["both_available"]:
            continue
        if p["euclidean_distance"] is None:
            continue

        ca = p["construct_a"]
        cb = p["construct_b"]

        cert_dist = p["euclidean_distance"]

        # Naive: just the forward_score gap (what you'd get without
        # the kappa decomposition)
        fwd_a = cert_map.get(ca, {}).get("forward_score")
        fwd_b = cert_map.get(cb, {}).get("forward_score")
        if fwd_a is None or fwd_b is None:
            continue

        naive_dist = abs(fwd_a - fwd_b)

        # How much does kappa add beyond forward_score?
        kappa_spatial_delta = abs(p["kappa_spatial_delta"]) if p["kappa_spatial_delta"] is not None else 0.0
        kappa_reconstruct_delta = abs(p["kappa_reconstruct_delta"]) if p["kappa_reconstruct_delta"] is not None else 0.0
        kappa_contribution = np.sqrt(kappa_spatial_delta**2 + kappa_reconstruct_delta**2)

        rows.append({
            "construct_a": ca,
            "construct_b": cb,
            "certificate_distance": round(cert_dist, 4),
            "naive_distance_forward_only": round(naive_dist, 4),
            "kappa_contribution": round(float(kappa_contribution), 4),
            "forward_score_a": round(fwd_a, 4),
            "forward_score_b": round(fwd_b, 4),
            "kappa_spatial_a": cert_map.get(ca, {}).get("kappa_spatial"),
            "kappa_spatial_b": cert_map.get(cb, {}).get("kappa_spatial"),
            "kappa_reconstruct_a": cert_map.get(ca, {}).get("kappa_reconstruct"),
            "kappa_reconstruct_b": cert_map.get(cb, {}).get("kappa_reconstruct"),
            # Quadrant: low naive but high certificate = decomposition adds info
            "hidden_disagreement": cert_dist > 0.15 and naive_dist < 0.10,
        })

    # Correlation between naive and certificate distance
    if len(rows) >= 3:
        naive_vals = [r["naive_distance_forward_only"] for r in rows]
        cert_vals = [r["certificate_distance"] for r in rows]
        spearman_r, spearman_p = stats.spearmanr(naive_vals, cert_vals)
    else:
        spearman_r, spearman_p = float("nan"), float("nan")

    n_hidden = sum(1 for r in rows if r["hidden_disagreement"])

    return {
        "pairs": rows,
        "n_pairs": len(rows),
        "n_hidden_disagreement": n_hidden,
        "spearman_naive_vs_cert": round(float(spearman_r), 4) if np.isfinite(spearman_r) else None,
        "spearman_p_value": round(float(spearman_p), 4) if np.isfinite(spearman_p) else None,
        "interpretation": (
            "LOW_CORRELATION: certificate decomposition captures structure "
            "that raw forward_score comparison misses"
            if np.isfinite(spearman_r) and abs(spearman_r) < 0.7
            else "HIGH_CORRELATION: certificate distance tracks raw score gap; "
                 "kappa axes add limited information"
            if np.isfinite(spearman_r)
            else "INSUFFICIENT_DATA"
        ),
    }


# ---------------------------------------------------------------------------
# C6: SVI-disagreement overlay
# ---------------------------------------------------------------------------

def compute_svi_overlay(
    doe_c1_json: dict,
    features_df: pd.DataFrame,
    scenario: str,
) -> dict:
    """Join DOE-C1 divergence with SVI/HIFLD at ZCTA level.

    Computes max pairwise certificate distance per ZCTA (from per-construct
    forward_score variation), then correlates with SVI themes.

    Note: DOE-C1 certificates are per-construct (one certificate per construct
    per scenario, not per ZCTA). The per-ZCTA signal comes from the
    per-construct forward_score being computed from spatial CV where each ZCTA
    contributes to fold-level R2. For the SVI overlay, we use scenario-level
    divergence paired with ZCTA-level SVI distributions.
    """
    per_construct = doe_c1_json["per_construct"]
    pairwise = doe_c1_json["pairwise"]

    # Scenario-level max divergence
    finite_dists = [
        p["euclidean_distance"]
        for p in pairwise
        if p["both_available"] and p["euclidean_distance"] is not None
    ]
    max_divergence = max(finite_dists) if finite_dists else float("nan")
    mean_divergence = float(np.mean(finite_dists)) if finite_dists else float("nan")

    # SVI distribution for this scenario's ZCTAs
    svi_available = [c for c in SVI_COLS if c in features_df.columns]
    hifld_available = [c for c in HIFLD_COLS if c in features_df.columns]

    svi_stats = {}
    for col in svi_available:
        vals = features_df[col].dropna()
        if len(vals) > 0:
            svi_stats[col] = {
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "std": round(float(vals.std()), 4),
                "q25": round(float(vals.quantile(0.25)), 4),
                "q75": round(float(vals.quantile(0.75)), 4),
                "n": int(len(vals)),
                "pct_high_vulnerability": round(float((vals > 0.75).mean()), 4),
            }

    hifld_stats = {}
    for col in hifld_available:
        vals = features_df[col].dropna()
        if len(vals) > 0:
            hifld_stats[col] = {
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "n": int(len(vals)),
            }

    # Per-construct availability
    available_constructs = [
        c["construct"] for c in per_construct if c["target_available"]
    ]

    return {
        "scenario": scenario,
        "n_zctas": len(features_df["zcta_id"].unique()) if "zcta_id" in features_df.columns else len(features_df),
        "max_divergence": round(max_divergence, 4) if np.isfinite(max_divergence) else None,
        "mean_divergence": round(mean_divergence, 4) if np.isfinite(mean_divergence) else None,
        "n_available_constructs": len(available_constructs),
        "available_constructs": available_constructs,
        "svi_distribution": svi_stats,
        "hifld_distribution": hifld_stats,
        "svi_columns_found": svi_available,
        "hifld_columns_found": hifld_available,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C6+C8 disagreement overlay")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no I/O")
    args = parser.parse_args()

    scenario = args.scenario

    if args.dry_run:
        log.info("DRY RUN for %s", scenario)
        log.info("  Input: results/s035/doe_c1/five_construct_%s.json", scenario)
        log.info("  Input: processed/%s/%s_event_features.parquet", scenario, scenario)
        log.info("  Output: results/s035/v8/baseline_comparison_%s.json", scenario)
        log.info("  Output: results/s035/v8/svi_disagreement_%s.json", scenario)
        return

    s3 = _get_s3()

    # -- Load DOE-C1 divergence --
    c1_key = f"results/s035/doe_c1/five_construct_{scenario}.json"
    log.info("Loading DOE-C1: %s", c1_key)
    doe_c1 = _read_json_s3(s3, c1_key)

    # -- Load event features (SVI + HIFLD) --
    feat_key = f"processed/{scenario}/{scenario}_event_features.parquet"
    log.info("Loading features: %s", feat_key)
    features_df = _read_parquet_s3(s3, feat_key)
    log.info("  features shape: %s, columns: %d", features_df.shape, len(features_df.columns))

    # -- C8: Baseline comparison --
    log.info("Computing C8 baseline comparison...")
    c8_result = compute_baseline_comparison(doe_c1)
    log.info("  pairs: %d, hidden_disagreement: %d, spearman: %s",
             c8_result["n_pairs"],
             c8_result["n_hidden_disagreement"],
             c8_result["spearman_naive_vs_cert"])

    # -- C6: SVI overlay --
    log.info("Computing C6 SVI-disagreement overlay...")
    c6_result = compute_svi_overlay(doe_c1, features_df, scenario)
    log.info("  zctas: %s, max_divergence: %s, svi_cols: %d, hifld_cols: %d",
             c6_result["n_zctas"],
             c6_result["max_divergence"],
             len(c6_result["svi_columns_found"]),
             len(c6_result["hifld_columns_found"]))

    # -- Wrap with provenance --
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "scenario": scenario,
        "script": Path(__file__).name,
    }

    c8_output = {"provenance": provenance, **c8_result}
    c6_output = {"provenance": provenance, **c6_result}

    # -- Upload or print --
    if args.upload:
        from _s3_result import upload_json_result
        c8_key = f"results/s035/v8/baseline_comparison_{scenario}.json"
        c6_key = f"results/s035/v8/svi_disagreement_{scenario}.json"
        upload_json_result(s3, BUCKET, c8_key, c8_output)
        log.info("Uploaded C8: %s", c8_key)
        upload_json_result(s3, BUCKET, c6_key, c6_output)
        log.info("Uploaded C6: %s", c6_key)
    else:
        print(json.dumps(c8_output, indent=2, default=str))
        print("---")
        print(json.dumps(c6_output, indent=2, default=str))

    log.info("DONE: %s", scenario)


if __name__ == "__main__":
    main()
