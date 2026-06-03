#!/usr/bin/env python3
"""
compute_geometry_kappa.py -- Phase 0.5: Geometry-only kappa per cell.

Computes kappa_geom from problem geometry BEFORE any model is trained.
kappa has zero computational dependency on RSN simplex values, fold
metrics, predictions, residuals, or model scores.  It measures geometric
compatibility (D*/D) -- how hard is this (scenario, target) cell given
the spatial structure, scale, and feature coverage.

Four geometry terms (all in [0, 1], higher = more compatible):

  1. spatial_connectivity  -- graph connectedness of scenario ZCTAs
  2. support_coverage       -- feature non-null coverage rate
  3. scale_stability        -- aggregation stability proxy (ZCTA vs county variance)
  4. administrative_alignment -- ZCTA-county boundary alignment (admin proxy; not hydrologic topology)

kappa_geom = mean(available terms), in [0, 1].

Inputs:  assembled parquet, adjacency edge list, crosswalk (all on S3)
Outputs: geometry_kappa.json on S3 (keyed by scenario x target)

Usage:
    python compute_geometry_kappa.py --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]
TARGETS = ["obs_nfip_event_claims", "obs_has_311", "obs_has_hwm"]

# Features used per arm (R0 static only -- geometry doesn't change per arm)
R0_FEATURES = [
    "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
    # QUARANTINED: county-level constants, zero within-scenario variance,
    # not temporally gated. Replaced by nfip_historical_frequency/severity.
    # "flood_event_count", "flood_event_count_5y", "flood_events_per_year",
    # "flood_property_damage_k", "flood_crop_damage_k",
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


# ---------------------------------------------------------------------------
# Geometry measurements (no model outputs, no predictions, no residuals)
# ---------------------------------------------------------------------------

def compute_spatial_connectivity(adj_df: pd.DataFrame,
                                  scenario_zctas: set[str]) -> float:
    """Graph connectedness: fraction of ZCTAs with at least one neighbor.

    Uses adjacency edge list only. No model outputs.
    Returns value in [0, 1]. 1 = fully connected, 0 = all islands.
    """
    if adj_df is None or adj_df.empty:
        return 0.0

    cols = adj_df.columns.tolist()
    src_col, dst_col = cols[0], cols[1]
    adj_df = adj_df.copy()
    adj_df[src_col] = adj_df[src_col].astype(str)
    adj_df[dst_col] = adj_df[dst_col].astype(str)

    # ZCTAs with at least one neighbor within scenario
    connected = set()
    for _, row in adj_df.iterrows():
        s, d = row[src_col], row[dst_col]
        if s in scenario_zctas and d in scenario_zctas:
            connected.add(s)
            connected.add(d)

    if not scenario_zctas:
        return 0.0
    return len(connected) / len(scenario_zctas)


def compute_support_coverage(df: pd.DataFrame, target: str,
                             features: list[str]) -> float:
    """Feature non-null coverage for rows where target is non-null.

    Measures: given the ZCTAs that have this target, what fraction of
    feature cells are populated?  Pure data inventory -- no model.
    Returns value in [0, 1]. 1 = all features present for all rows.
    """
    target_mask = df[target].notna() if target in df.columns else pd.Series(False, index=df.index)
    sub = df.loc[target_mask, [f for f in features if f in df.columns]]
    if sub.empty:
        return 0.0
    return float(sub.notna().mean().mean())


def compute_scale_stability(df: pd.DataFrame, crosswalk: pd.DataFrame,
                            features: list[str]) -> float:
    """Aggregation stability proxy: between-county / total variance ratio.

    High value = feature structure aligns with county aggregation.
    Low value = county boundaries do not capture feature variation.
    This is a proxy for aggregation sensitivity, not full MAUP testing.
    Uses feature values and county assignments only. No model outputs.
    """
    if crosswalk is None or crosswalk.empty:
        return 0.5  # Neutral if no crosswalk

    df = df.copy()
    df["zcta_id"] = df["zcta_id"].astype(str)
    xwalk = crosswalk.copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)

    merged = df.merge(
        xwalk[["zcta_id", "county_fips"]].drop_duplicates(),
        on="zcta_id", how="left",
    )

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
        county_means = vals.groupby("county_fips")[col].mean()
        between_var = county_means.var()
        # High between/total = county structure explains variance = scale-stable
        ratios.append(between_var / total_var)

    if not ratios:
        return 0.5
    return float(np.clip(np.mean(ratios), 0.0, 1.0))


def compute_administrative_alignment(crosswalk: pd.DataFrame,
                                     scenario_zctas: set[str]) -> float:
    """Administrative boundary alignment: 1 - fraction of multi-county ZCTAs.

    Multi-county ZCTAs indicate postal boundaries cross jurisdictions,
    creating ambiguity in blocking and spatial attribution.
    This is an administrative proxy; future versions may supplement
    with hydrologic topology (catchments, basins, flow paths).
    Uses crosswalk only. No model outputs.
    """
    if crosswalk is None or crosswalk.empty:
        return 0.5

    xwalk = crosswalk.copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
    xwalk = xwalk[xwalk["zcta_id"].isin(scenario_zctas)]

    if xwalk.empty:
        return 0.5

    counties_per_zcta = xwalk.groupby("zcta_id")["county_fips"].nunique()
    multi = (counties_per_zcta > 1).sum()
    return 1.0 - (multi / len(counties_per_zcta))


# ---------------------------------------------------------------------------
# Independence guard: kappa must not read model-produced artifacts
# ---------------------------------------------------------------------------

# S3 prefixes that contain model outputs, predictions, diagnostics,
# certificates, or any artifact produced after training.  If this script
# ever reads from one of these, it is a kappa independence violation.
FORBIDDEN_S3_PREFIXES = (
    # Model training outputs
    "results/s035/r0_",
    "results/s035/r1_",
    "results/s035/r2_",
    # Model-derived diagnostics and certificates
    "results/s035/diagnostics_",
    "results/s035/certificates_",
    # Uplift / money table (derived from model comparisons)
    "results/s035/uplift_",
    "results/s035/money_table",
    # Predictions and residuals
    "results/s035/predictions",
    "results/s035/residuals",
    # Fold assignments (training artifact)
    "folds/",
)


def _guard_s3_read(key: str) -> None:
    """Raise if an S3 key belongs to model-produced output."""
    for prefix in FORBIDDEN_S3_PREFIXES:
        if key.startswith(prefix):
            raise RuntimeError(
                f"KAPPA INDEPENDENCE VIOLATION: compute_geometry_kappa.py "
                f"attempted to read '{key}', which is a model-produced "
                f"artifact.  kappa must have zero dependency on model "
                f"training, validation, predictions, or diagnostics."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_geometry_kappa(s3) -> dict:
    """Compute kappa_geom for all (scenario, target) cells."""
    cells = []

    for scenario in MODELABLE:
        log.info("Computing geometry kappa for %s", scenario)

        # Load geometry-only inputs
        try:
            df = load_processed_parquet(s3, scenario)
            df["zcta_id"] = df["zcta_id"].astype(str)
        except Exception as e:
            log.warning("Cannot load parquet for %s: %s", scenario, e)
            continue

        scenario_zctas = set(df["zcta_id"].unique())

        # Adjacency (optional but important)
        adj_df = None
        for adj_key in [
            "raw/geocertdb2026/zcta_adjacency.parquet",
            "raw/geocert/zcta_adjacency.parquet",
        ]:
            _guard_s3_read(adj_key)
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=adj_key)
                adj_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
                break
            except Exception:
                continue

        # Crosswalk (optional)
        try:
            crosswalk = load_crosswalk(s3)
        except Exception:
            crosswalk = None

        # Geometry terms (computed once per scenario, target-independent)
        frag = compute_spatial_connectivity(adj_df, scenario_zctas)
        scale = compute_scale_stability(df, crosswalk, R0_FEATURES)
        topo = compute_administrative_alignment(crosswalk, scenario_zctas)

        for target in TARGETS:
            if target not in df.columns:
                continue
            if df[target].notna().sum() == 0:
                continue

            coverage = compute_support_coverage(df, target, R0_FEATURES)

            terms = {
                "spatial_connectivity": round(frag, 6),
                "support_coverage": round(coverage, 6),
                "scale_stability": round(scale, 6),
                "administrative_alignment": round(topo, 6),
            }
            available = [v for v in terms.values() if v is not None]
            kappa_geom = float(np.mean(available)) if available else 0.0

            cell = {
                "scenario": scenario,
                "target": target,
                "kappa_geom": round(kappa_geom, 6),
                **terms,
                "n_zctas": len(scenario_zctas),
                "n_target_nonull": int(df[target].notna().sum()),
            }
            cells.append(cell)

            log.info(
                "  %s / %s: kappa_geom=%.3f (frag=%.3f cov=%.3f scale=%.3f topo=%.3f)",
                scenario, target, kappa_geom, frag, coverage, scale, topo,
            )

    return {
        "experiment": "s035-model-ladder",
        "phase": "geometry_kappa",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Geometry-only kappa. Zero dependency on RSN simplex, fold "
            "metrics, predictions, residuals, or model scores."
        ),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0.5: Geometry-only kappa (pre-training)"
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  S035 PHASE 0.5: GEOMETRY KAPPA (pre-training)")
    print("=" * 60 + "\n")

    payload = compute_geometry_kappa(s3)

    # Always write locally
    local_path = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    local_path.mkdir(parents=True, exist_ok=True)
    with open(local_path / "geometry_kappa.json", "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote local: %s", local_path / "geometry_kappa.json")

    if args.upload:
        key = f"{RESULTS_PREFIX}/geometry_kappa.json"
        upload_json_result(s3, BUCKET, key, payload)

    # Summary
    n = len(payload["cells"])
    if n:
        kappas = [c["kappa_geom"] for c in payload["cells"]]
        print(f"\nGeometry kappa: {n} cells, range [{min(kappas):.3f}, {max(kappas):.3f}]")
    else:
        print("\nNo cells computed.")


if __name__ == "__main__":
    main()
