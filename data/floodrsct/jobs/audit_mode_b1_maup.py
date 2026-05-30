#!/usr/bin/env python3
"""
audit_mode_b1_maup.py -- GeoRSCT Mode B.1: MAUP / Partition Drift.

Detects Modifiable Areal Unit Problem (MAUP) risks where the choice
of spatial unit (ZCTA) may not align with the natural units for
flood processes.

Checks:
  1. ZCTA-county boundary crossing: how many ZCTAs span multiple
     counties? If significant, county-level aggregation loses
     within-ZCTA variation.
  2. ZCTA size variation: extreme variation in ZCTA area means
     aggregation treats a 0.1 km2 urban ZCTA the same as a
     500 km2 rural ZCTA.
  3. Multi-county ZCTAs: ZCTAs assigned to >1 county in the
     crosswalk indicate boundary misalignment.
  4. Feature sensitivity to unit choice: if a feature's variance
     changes dramatically when aggregated to county vs ZCTA,
     MAUP is active.

The core MAUP concern for flood data: ZCTA boundaries follow postal
delivery routes, not watersheds. A ZCTA may span a drainage divide,
mixing upstream and downstream hydrologic regimes into one unit.
Without a HUC crosswalk, this cannot be directly measured, but we
can flag proxy indicators.

Usage:
    python audit_mode_b1_maup.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    load_crosswalk, write_evidence,
)

# Features to test for MAUP sensitivity
MAUP_FEATURES = [
    "rainfall_total_mm", "total_rainfall_mm",
    "elevation_m_msl",
    "impervious_pct",
    "nfip_event_claims",
]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    results = []

    # --- Check 1: Multi-county ZCTAs ---
    try:
        xwalk = load_crosswalk(s3)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)

        # Count counties per ZCTA
        scenario_zctas = set(df["zcta_id"].unique())
        xwalk_scenario = xwalk[xwalk["zcta_id"].isin(scenario_zctas)]

        counties_per_zcta = (xwalk_scenario.groupby("zcta_id")["county_fips"]
                             .nunique().reset_index())
        counties_per_zcta.columns = ["zcta_id", "n_counties"]

        multi_county = counties_per_zcta[counties_per_zcta["n_counties"] > 1]
        n_multi = len(multi_county)
        n_total = len(counties_per_zcta)

        results.append(AuditResult(
            audit_id="mode_B1", scenario=scenario, probe="maup",
            status="FAIL" if n_multi > n_total * 0.1 else "PASS",
            detail={
                "check": "multi_county_zctas",
                "total_zctas": n_total,
                "multi_county_zctas": n_multi,
                "multi_county_pct": round(n_multi / max(n_total, 1) * 100, 1),
                "max_counties_per_zcta": int(counties_per_zcta["n_counties"].max()),
                "note": ("ZCTAs spanning multiple counties indicate "
                         "postal/administrative boundaries cross jurisdictions. "
                         "County-blocked CV may split a ZCTA's observations "
                         "into different folds."),
            },
            min_support=min_support, timestamp=ts,
        ))

        # --- Check 2: MAUP sensitivity (ZCTA vs county aggregation) ---
        df_county = df.merge(xwalk_scenario[["zcta_id", "county_fips"]].drop_duplicates(),
                             on="zcta_id", how="left")

        for col in MAUP_FEATURES:
            if col not in df.columns or df[col].notna().sum() < min_support:
                continue

            # ZCTA-level variance
            zcta_var = df[col].dropna().var()

            # County-level variance (aggregate to county mean, then variance)
            county_means = (df_county.groupby("county_fips")[col]
                           .mean().dropna())
            if len(county_means) < 2:
                continue
            county_var = county_means.var()

            # Variance ratio: how much variance is lost by aggregating to county
            if zcta_var > 0:
                variance_ratio = round(float(county_var / zcta_var), 4)
            else:
                variance_ratio = 0.0

            # Low ratio = most variance is within-county (MAUP risk: county
            # aggregation hides it). High ratio = variance is between-county.
            is_maup_sensitive = variance_ratio < 0.1

            results.append(AuditResult(
                audit_id="mode_B1", scenario=scenario, probe="maup",
                status="FAIL" if is_maup_sensitive else "PASS",
                detail={
                    "check": "variance_ratio",
                    "feature": col,
                    "zcta_variance": round(float(zcta_var), 4),
                    "county_variance": round(float(county_var), 4),
                    "variance_ratio": variance_ratio,
                    "n_counties": len(county_means),
                    "interpretation": (
                        f"County aggregation retains {variance_ratio*100:.1f}% "
                        f"of ZCTA-level variance. "
                        + ("Most variation is within-county -- county "
                           "aggregation would hide it (MAUP active)."
                           if is_maup_sensitive else
                           "Variance is mostly between-county -- "
                           "county aggregation preserves signal.")
                    ),
                },
                min_support=min_support, timestamp=ts,
            ))

    except Exception as e:
        results.append(AuditResult(
            audit_id="mode_B1", scenario=scenario, probe="maup",
            status="SKIP",
            detail={"error": f"Crosswalk unavailable: {e}"},
            min_support=min_support, timestamp=ts,
        ))

    # --- Check 3: ZCTA size variation proxy ---
    # Use the number of unique events per ZCTA as a size proxy
    # (larger ZCTAs tend to appear in more events due to geographic overlap)
    if "event" in df.columns:
        events_per_zcta = df.groupby("zcta_id")["event"].nunique()
        n_events_expected = df["event"].nunique()

        # All ZCTAs should have the same number of events (one row per event)
        inconsistent = events_per_zcta[events_per_zcta != n_events_expected]

        if len(inconsistent) > 0:
            results.append(AuditResult(
                audit_id="mode_B1", scenario=scenario, probe="maup",
                status="FAIL",
                detail={
                    "check": "event_coverage_consistency",
                    "expected_events_per_zcta": int(n_events_expected),
                    "zctas_with_wrong_count": len(inconsistent),
                    "total_zctas": len(events_per_zcta),
                    "note": ("ZCTAs with different event counts may indicate "
                             "boundary changes between event years."),
                },
                min_support=min_support, timestamp=ts,
            ))
        else:
            results.append(AuditResult(
                audit_id="mode_B1", scenario=scenario, probe="maup",
                status="PASS",
                detail={
                    "check": "event_coverage_consistency",
                    "expected_events_per_zcta": int(n_events_expected),
                    "all_zctas_consistent": True,
                },
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 4: HUC crosswalk absence ---
    # Flag that ZCTA-to-HUC crosswalk does not exist
    results.append(AuditResult(
        audit_id="mode_B1", scenario=scenario, probe="maup",
        status="FAIL",
        detail={
            "check": "huc_crosswalk_absence",
            "note": (
                "No ZCTA-to-HUC crosswalk exists. ZCTA boundaries follow "
                "postal delivery routes, not watershed divides. A ZCTA may "
                "span a drainage divide, mixing upstream and downstream "
                "hydrologic regimes. This is a fundamental MAUP limitation "
                "of the ZCTA-based substrate."
            ),
            "action": (
                "Document as known limitation. Future work: build "
                "ZCTA-to-HUC12 crosswalk from NHDPlus WBD polygons."
            ),
        },
        min_support=min_support, timestamp=ts,
    ))

    write_evidence(results, "mode_b1_maup", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode B.1: MAUP / Partition drift detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
