#!/usr/bin/env python3
"""
compute_rsct_certificates.py -- RSCT certificate generation for GeoRSCT.

Three modes:
  --collapse    Eval-geometry score-collapse demo: show how R/S/N simplex
                collapses when geometry kappa degrades.
  --certificate Certificate sidecar generator: produce ReadinessCertificate
                JSON per (scenario, target) cell.
  --sweep       Inflation-vs-sigma sweep: vary inflation factor and measure
                sigma_geo (turbulence) across all 27 (scenario, target) cells.

All three modes call the same build_certificate() pipeline:
  1. Load scenario parquet + adjacency + crosswalk from S3
  2. Compute geometry kappa (spatial_connectivity, support_coverage,
     scale_stability, admin_alignment)
  3. Run Ridge CV to get fold metrics -> R, S_sup, N
  4. Compute sigma_geo from fold metric std
  5. Build ReadinessCertificate with oobleck-calibrated kappa_compat

Inputs:  assembled parquets, adjacency, crosswalk, folds (all on S3)
Outputs: JSON results uploaded to s3://swarm-floodrsct-data/results/s035/

Usage:
    python compute_rsct_certificates.py --collapse --upload
    python compute_rsct_certificates.py --certificate --upload
    python compute_rsct_certificates.py --sweep --upload
"""

import argparse
import io
import json
import logging
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
from _s3_result import upload_json_result
from generate_folds import generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SEED = 42

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

TARGETS = [
    ("obs_nfip_event_claims", "regression", "log1p"),
    ("obs_has_311",           "classification", None),
    ("obs_has_hwm",           "classification", None),
]

R0_FEATURES = [
    "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
    "elevation_m_msl", "slope_mean_pct", "twi_twi",
    "coastal_distance_m", "latitude", "longitude",
    "acs_total_pop", "acs_median_hh_income", "acs_pct_below_poverty",
    "acs_pct_renter_occupied", "acs_pct_owner_occupied", "acs_pct_vacant",
    "acs_pct_no_vehicle", "acs_median_home_value", "acs_median_year_built",
    "svi_overall", "svi_socioeconomic", "svi_household_disability",
    "svi_minority_language", "svi_housing_transport",
    "nfip_historical_frequency", "nfip_historical_severity",
    "hifld_nearest_hospital_km", "hifld_n_hospitals", "population",
    "impervious_pct", "cropland_pct",
]

# Oobleck threshold parameters (ADR-023: sigmoidal, not linear)
KAPPA_BASE = 0.4
DELTA_KAPPA = 0.3
STEEPNESS = 8.0
SIGMA_C = 0.3


# ---------------------------------------------------------------------------
# Geometry measurements
# ---------------------------------------------------------------------------

def compute_spatial_connectivity(adj_df: pd.DataFrame,
                                  scenario_zctas: set) -> float:
    """Fraction of ZCTAs with at least one neighbor."""
    if adj_df is None or adj_df.empty:
        return 0.0
    cols = adj_df.columns.tolist()
    src, dst = cols[0], cols[1]
    connected = set()
    for _, row in adj_df.iterrows():
        s, d = str(row[src]), str(row[dst])
        if s in scenario_zctas and d in scenario_zctas:
            connected.update([s, d])
    return len(connected) / max(len(scenario_zctas), 1)


def compute_support_coverage(df: pd.DataFrame, target: str,
                              features: list) -> float:
    """Feature non-null rate for rows where target is present."""
    if target not in df.columns:
        return 0.0
    sub = df.loc[df[target].notna(), [f for f in features if f in df.columns]]
    if sub.empty:
        return 0.0
    return float(sub.notna().mean().mean())


def compute_scale_stability(df: pd.DataFrame, crosswalk: pd.DataFrame,
                             features: list) -> float:
    """Between-county / total variance ratio."""
    if crosswalk is None or crosswalk.empty:
        return 0.5
    df2 = df.copy()
    df2["zcta_id"] = df2["zcta_id"].astype(str)
    xwalk = crosswalk[["zcta_id", "county_fips"]].drop_duplicates().copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
    merged = df2.merge(xwalk, on="zcta_id", how="left")
    ratios = []
    for col in features:
        if col not in merged.columns:
            continue
        vals = merged[[col, "county_fips"]].dropna()
        if len(vals) < 10:
            continue
        total_var = vals[col].var()
        if total_var < 1e-12:
            continue
        between_var = vals.groupby("county_fips")[col].mean().var()
        ratios.append(between_var / total_var)
    return float(np.clip(np.mean(ratios), 0.0, 1.0)) if ratios else 0.5


def compute_admin_alignment(df: pd.DataFrame, crosswalk: pd.DataFrame) -> float:
    """1 - fraction of ZCTAs spanning multiple counties."""
    if crosswalk is None or crosswalk.empty:
        return 0.5
    xwalk = crosswalk[["zcta_id", "county_fips"]].drop_duplicates()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
    zctas_in_scenario = set(df["zcta_id"].astype(str))
    xwalk = xwalk[xwalk["zcta_id"].isin(zctas_in_scenario)]
    multi = xwalk.groupby("zcta_id")["county_fips"].nunique()
    frac_multi = (multi > 1).mean()
    return float(1.0 - frac_multi)


def compute_kappa_geom(spatial_conn: float, support_cov: float,
                        scale_stab: float, admin_align: float) -> float:
    """Mean of four geometry terms."""
    return float(np.mean([spatial_conn, support_cov, scale_stab, admin_align]))


# ---------------------------------------------------------------------------
# Oobleck threshold (ADR-023: sigmoidal)
# ---------------------------------------------------------------------------

def oobleck_kappa_req(sigma: float) -> float:
    """Sigmoidal kappa requirement as function of turbulence."""
    return KAPPA_BASE + DELTA_KAPPA / (1.0 + math.exp(-STEEPNESS * (sigma - SIGMA_C)))


# ---------------------------------------------------------------------------
# RSN simplex from fold metrics
# ---------------------------------------------------------------------------

def compute_rsn_from_folds(fold_scores: list) -> dict:
    """Compute R, S_sup, N from cross-validation fold scores.

    R = clipped mean score (relevance proxy)
    S_sup = 1 - CV (stability proxy)
    N = 1 - R - S_sup (noise proxy)
    sigma = std of fold scores
    """
    arr = np.array(fold_scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return {"R": 0.0, "S_sup": 0.0, "N": 1.0, "sigma": 1.0, "n_folds": 0}

    mu = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    cv = std / max(abs(mu), 1e-12)

    # Clamp to simplex
    R = float(np.clip(mu, 0.0, 1.0))
    S_sup = float(np.clip(1.0 - cv, 0.0, 1.0 - R))
    N = float(np.clip(1.0 - R - S_sup, 0.0, 1.0))

    return {"R": R, "S_sup": S_sup, "N": N, "sigma": std, "n_folds": len(arr)}


# ---------------------------------------------------------------------------
# Certificate builder
# ---------------------------------------------------------------------------

def build_certificate(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    transform: str,
    adj_df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    scenario: str,
    inflation: float = 1.0,
) -> dict:
    """Build a ReadinessCertificate for one (scenario, target) cell.

    Args:
        inflation: Multiplicative factor applied to sigma for sweep mode.
                   1.0 = no inflation.
    """
    features = [f for f in R0_FEATURES if f in df.columns]
    zctas = set(df["zcta_id"].astype(str))

    # Geometry kappa
    sp_conn = compute_spatial_connectivity(adj_df, zctas)
    sup_cov = compute_support_coverage(df, target, features)
    sc_stab = compute_scale_stability(df, crosswalk, features)
    ad_algn = compute_admin_alignment(df, crosswalk)
    kappa_geom = compute_kappa_geom(sp_conn, sup_cov, sc_stab, ad_algn)

    # Prepare data for Ridge CV
    if target not in df.columns:
        return _missing_cert(scenario, target, "target column missing", kappa_geom)

    sub = df[features + [target]].dropna()
    if len(sub) < 20:
        return _missing_cert(scenario, target, "insufficient data (n<%d)" % len(sub), kappa_geom)

    X = sub[features].values
    y = sub[target].values.copy()

    if transform == "log1p":
        y = np.log1p(np.clip(y, 0, None))

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Ridge CV with 5-fold
    scoring = "r2" if task_type == "regression" else "roc_auc"
    try:
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        fold_scores = cross_val_score(ridge, X, y, cv=5, scoring=scoring,
                                       n_jobs=-1)
    except Exception as e:
        return _missing_cert(scenario, target, "CV failed: %s" % str(e), kappa_geom)

    # RSN from folds
    rsn = compute_rsn_from_folds(fold_scores.tolist())
    sigma = rsn["sigma"] * inflation

    # kappa_compat = R*(1-N), clamped
    kappa_compat = float(np.clip(rsn["R"] * (1.0 - rsn["N"]), 0.0, 1.0))

    # Oobleck threshold
    kappa_req = oobleck_kappa_req(sigma)

    # Public decision
    if rsn["R"] < 0.3:
        decision = "REFUSE"
    elif kappa_compat < kappa_req:
        decision = "CAUTION"
    else:
        decision = "EXECUTE"

    return {
        "scenario": scenario,
        "target": target,
        "task_type": task_type,
        "transform": transform,
        "n_samples": len(sub),
        "n_features": len(features),
        "fold_scores": fold_scores.tolist(),
        "R": rsn["R"],
        "S_sup": rsn["S_sup"],
        "N": rsn["N"],
        "sigma": sigma,
        "inflation": inflation,
        "kappa_compat": kappa_compat,
        "kappa_geom": kappa_geom,
        "kappa_req": kappa_req,
        "public_decision": decision,
        "geometry": {
            "spatial_connectivity": sp_conn,
            "support_coverage": sup_cov,
            "scale_stability": sc_stab,
            "admin_alignment": ad_algn,
        },
        "simplex_valid": abs(rsn["R"] + rsn["S_sup"] + rsn["N"] - 1.0) < 1e-6,
    }


def _missing_cert(scenario: str, target: str, reason: str,
                   kappa_geom: float) -> dict:
    """Return a certificate stub for missing/insufficient data."""
    return {
        "scenario": scenario,
        "target": target,
        "R": 0.0, "S_sup": 0.0, "N": 1.0,
        "sigma": float("nan"),
        "kappa_compat": 0.0,
        "kappa_geom": kappa_geom,
        "kappa_req": float("nan"),
        "public_decision": "REFUSE",
        "missing_reason": reason,
        "simplex_valid": True,
    }


# ---------------------------------------------------------------------------
# Mode: --collapse
# ---------------------------------------------------------------------------

def run_collapse(s3, upload: bool) -> None:
    """Score-collapse demo: show R/S/N simplex vs geometry kappa."""
    log.info("=== COLLAPSE MODE ===")
    crosswalk = load_crosswalk(s3)
    adj_df = _load_adjacency(s3)

    results = []
    for scenario in MODELABLE:
        df = load_processed_parquet(s3, scenario)
        for target, task_type, transform in TARGETS:
            cert = build_certificate(df, target, task_type, transform,
                                      adj_df, crosswalk, scenario)
            results.append(cert)
            log.info(
                "%s / %s: R=%.3f S_sup=%.3f N=%.3f kappa_geom=%.3f "
                "kappa_compat=%.3f -> %s",
                scenario, target,
                cert["R"], cert["S_sup"], cert["N"],
                cert.get("kappa_geom", 0),
                cert.get("kappa_compat", 0),
                cert["public_decision"],
            )

    # Sort by kappa_geom to show collapse gradient
    results.sort(key=lambda c: c.get("kappa_geom", 0))

    payload = {
        "mode": "collapse",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Score-collapse demo: R/S/N vs geometry kappa gradient",
        "n_cells": len(results),
        "cells": results,
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_collapse.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Mode: --certificate
# ---------------------------------------------------------------------------

def run_certificate(s3, upload: bool) -> None:
    """Generate ReadinessCertificate sidecar per (scenario, target)."""
    log.info("=== CERTIFICATE MODE ===")
    crosswalk = load_crosswalk(s3)
    adj_df = _load_adjacency(s3)

    all_certs = {}
    for scenario in MODELABLE:
        df = load_processed_parquet(s3, scenario)
        scenario_certs = []
        for target, task_type, transform in TARGETS:
            cert = build_certificate(df, target, task_type, transform,
                                      adj_df, crosswalk, scenario)
            scenario_certs.append(cert)
            log.info(
                "CERT %s/%s: decision=%s R=%.3f kappa_compat=%.3f",
                scenario, target, cert["public_decision"],
                cert["R"], cert.get("kappa_compat", 0),
            )
        all_certs[scenario] = scenario_certs

    payload = {
        "mode": "certificate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "ReadinessCertificate sidecar per (scenario, target)",
        "scenarios": all_certs,
        "summary": {
            "total_cells": sum(len(v) for v in all_certs.values()),
            "execute": sum(
                1 for certs in all_certs.values()
                for c in certs if c["public_decision"] == "EXECUTE"
            ),
            "caution": sum(
                1 for certs in all_certs.values()
                for c in certs if c["public_decision"] == "CAUTION"
            ),
            "refuse": sum(
                1 for certs in all_certs.values()
                for c in certs if c["public_decision"] == "REFUSE"
            ),
        },
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_certificates.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Mode: --sweep
# ---------------------------------------------------------------------------

def run_sweep(s3, upload: bool) -> None:
    """Inflation-vs-sigma sweep across all cells."""
    log.info("=== SWEEP MODE ===")
    crosswalk = load_crosswalk(s3)
    adj_df = _load_adjacency(s3)

    inflations = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    sweep_results = []

    for scenario in MODELABLE:
        df = load_processed_parquet(s3, scenario)
        for target, task_type, transform in TARGETS:
            row = {"scenario": scenario, "target": target, "points": []}
            for inf in inflations:
                cert = build_certificate(
                    df, target, task_type, transform,
                    adj_df, crosswalk, scenario,
                    inflation=inf,
                )
                row["points"].append({
                    "inflation": inf,
                    "sigma": cert.get("sigma"),
                    "kappa_compat": cert.get("kappa_compat"),
                    "kappa_req": cert.get("kappa_req"),
                    "R": cert["R"],
                    "S_sup": cert["S_sup"],
                    "N": cert["N"],
                    "public_decision": cert["public_decision"],
                })
            sweep_results.append(row)
            base = row["points"][2]  # inflation=1.0
            log.info(
                "SWEEP %s/%s: base sigma=%.4f, decisions=%s",
                scenario, target,
                base.get("sigma", float("nan")),
                [p["public_decision"] for p in row["points"]],
            )

    payload = {
        "mode": "sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Inflation-vs-sigma_geo sweep across 15 (scenario, target) cells",
        "inflations": inflations,
        "n_cells": len(sweep_results),
        "cells": sweep_results,
        "oobleck_params": {
            "kappa_base": KAPPA_BASE,
            "delta_kappa": DELTA_KAPPA,
            "steepness": STEEPNESS,
            "sigma_c": SIGMA_C,
        },
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_sweep.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Shared data loaders
# ---------------------------------------------------------------------------

def _load_adjacency(s3) -> pd.DataFrame:
    """Load ZCTA adjacency edge list from S3."""
    for key in [
        "raw/geocertdb2026/zcta_adjacency.parquet",
        "raw/geocert/zcta_adjacency.parquet",
    ]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            return pd.read_parquet(buf)
        except Exception:
            continue
    log.warning("zcta_adjacency.parquet not found on S3; connectivity will be 0")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSCT certificate generation for GeoRSCT"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collapse", action="store_true",
                      help="Score-collapse demo")
    mode.add_argument("--certificate", action="store_true",
                      help="Certificate sidecar generator")
    mode.add_argument("--sweep", action="store_true",
                      help="Inflation-vs-sigma sweep")

    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit")
    args = parser.parse_args()

    if args.dry_run:
        log.info("[DRY RUN] Mode: %s",
                 "collapse" if args.collapse else
                 "certificate" if args.certificate else "sweep")
        log.info("[DRY RUN] Scenarios: %s", MODELABLE)
        log.info("[DRY RUN] Targets: %s", [t[0] for t in TARGETS])
        log.info("[DRY RUN] Upload: %s", args.upload)
        return

    s3 = get_s3_client()

    if args.collapse:
        run_collapse(s3, args.upload)
    elif args.certificate:
        run_certificate(s3, args.upload)
    elif args.sweep:
        run_sweep(s3, args.upload)

    log.info("Done.")


if __name__ == "__main__":
    main()
