#!/usr/bin/env python3
"""
audit_mode_c2_crs.py -- GeoRSCT Mode C.2: Projection / CRS Inconsistency.

Detects silent CRS mismatches that cause wrong-pixel sampling. The
canonical failure mode: NLCD is natively EPSG:5070 (Albers Equal Area
Conic). If sampled as EPSG:4326 (lat/lon), centroid coordinates index
wrong pixels, producing garbage impervious_pct values.

Checks:
  1. NLCD impervious_pct: values should be 0-100. If reprojection
     failed silently, values may be constant (always sampling the
     same pixel), all-zero (sampling ocean), or have physically
     implausible distributions (e.g., desert ZCTAs showing >80%).
  2. DEM elevation: NAVD88 vs WGS84 ellipsoidal height confusion
     causes ~20-40 m offsets. Coastal ZCTAs with negative elevation
     under one datum may be positive under another.
  3. SLOSH MOM: local SLOSH basin projections must be correctly
     transformed to match ZCTA centroids in EPSG:4326.
  4. MRMS: HRAP grid coordinates must be decoded correctly from
     cfgrib. The 1D-to-2D meshgrid expansion is a known failure
     point (see METHODS.md).

Detection strategy:
  - Constant-value features suggest single-pixel sampling (CRS offset
    maps all centroids to one grid cell)
  - Physically implausible values for known geography (e.g., high
    impervious in rural/agricultural ZCTAs)
  - Elevation sign flips at coast (negative when should be positive)

Usage:
    python audit_mode_c2_crs.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    write_evidence,
)

# Scenarios where each CRS-sensitive feature appears
NLCD_SCENARIOS = ["houston", "nyc"]
DEM_SCENARIOS = ["southwest_florida", "new_orleans"]
SLOSH_SCENARIOS = ["southwest_florida"]


def _check_constant_or_degenerate(values, col_name: str) -> dict:
    """Check if a feature is constant, near-constant, or degenerate."""
    n = len(values)
    n_unique = len(np.unique(values))
    result = {
        "column": col_name,
        "n_values": n,
        "n_unique": n_unique,
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
    }

    if n_unique == 1:
        result["flag"] = "CONSTANT"
        result["interpretation"] = (
            "All values identical -- likely single-pixel sampling "
            "from CRS offset"
        )
    elif n_unique <= 3 and n > 20:
        result["flag"] = "NEAR_CONSTANT"
        result["interpretation"] = (
            f"Only {n_unique} distinct values across {n} rows -- "
            "possible coarse quantization or CRS misalignment"
        )
    else:
        result["flag"] = "OK"

    return result


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    # --- Check 1: NLCD impervious_pct CRS integrity ---
    if scenario in NLCD_SCENARIOS and "impervious_pct" in df.columns:
        values = df["impervious_pct"].dropna().values

        if len(values) > 0:
            check = _check_constant_or_degenerate(values, "impervious_pct")
            check["check"] = "nlcd_crs"
            check["expected_crs"] = "EPSG:5070 (Albers Equal Area Conic)"
            check["risk"] = (
                "NLCD natively uses EPSG:5070. Sampling with EPSG:4326 "
                "coordinates without reprojection reads wrong pixels."
            )

            # Physical plausibility: impervious should be 0-100
            out_of_range = int(np.sum((values < 0) | (values > 100)))
            check["out_of_range_count"] = out_of_range

            # All-zero check (sampling ocean/outside CONUS)
            all_zero = int(np.sum(values == 0))
            check["zero_count"] = all_zero
            check["zero_pct"] = round(all_zero / len(values) * 100, 1)

            is_bad = (check["flag"] != "OK" or out_of_range > 0 or
                      check["zero_pct"] > 90)
            status = "FAIL" if is_bad else "PASS"

            results.append(AuditResult(
                audit_id="mode_C2", scenario=scenario, probe="crs",
                status=status, detail=check,
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 2: DEM elevation datum integrity ---
    if scenario in DEM_SCENARIOS and "elevation_m_msl" in df.columns:
        values = df["elevation_m_msl"].dropna().values

        if len(values) > 0:
            check = _check_constant_or_degenerate(values, "elevation_m_msl")
            check["check"] = "dem_datum"
            check["expected_datum"] = "NAVD88"
            check["risk"] = (
                "NAVD88 vs WGS84 ellipsoidal height confusion causes "
                "~20-40 m offsets. Coastal ZCTAs may flip sign."
            )

            # Coastal scenarios: check for implausible negatives
            n_negative = int(np.sum(values < -10))
            check["n_below_minus10m"] = n_negative

            # Check for implausible positives (> 500m for FL/LA coast)
            n_high = int(np.sum(values > 500))
            check["n_above_500m"] = n_high

            is_bad = (check["flag"] != "OK" or n_negative > 0 or
                      n_high > len(values) * 0.1)
            status = "FAIL" if is_bad else "PASS"

            results.append(AuditResult(
                audit_id="mode_C2", scenario=scenario, probe="crs",
                status=status, detail=check,
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 3: SLOSH MOM projection integrity ---
    if scenario in SLOSH_SCENARIOS and "slosh_max_surge_m" in df.columns:
        values = df["slosh_max_surge_m"].dropna().values

        if len(values) > 0:
            check = _check_constant_or_degenerate(values, "slosh_max_surge_m")
            check["check"] = "slosh_projection"
            check["risk"] = (
                "SLOSH basins use local projections. GeoTIFF must be "
                "correctly georeferenced for ZCTA centroid sampling."
            )

            # Physically implausible: surge > 20m is extreme
            n_extreme = int(np.sum(values > 20))
            check["n_above_20m"] = n_extreme

            # Negative surge is nonsensical for MOM inundation
            n_negative = int(np.sum(values < 0))
            check["n_negative"] = n_negative

            is_bad = (check["flag"] != "OK" or n_extreme > 0 or
                      n_negative > 0)
            status = "FAIL" if is_bad else "PASS"

            results.append(AuditResult(
                audit_id="mode_C2", scenario=scenario, probe="crs",
                status=status, detail=check,
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 4: MRMS rainfall grid integrity ---
    rain_col = None
    for c in ["rainfall_total_mm", "total_rainfall_mm"]:
        if c in df.columns:
            rain_col = c
            break

    if rain_col:
        values = df[rain_col].dropna().values

        if len(values) > 0:
            check = _check_constant_or_degenerate(values, rain_col)
            check["check"] = "mrms_grid"
            check["risk"] = (
                "cfgrib returns 1D coordinate arrays. Meshgrid expansion "
                "must use np.meshgrid(lon, lat) not (lat, lon) or all "
                "centroids map to wrong grid cells."
            )

            # Physically implausible: > 2000mm in one event
            n_extreme = int(np.sum(values > 2000))
            check["n_above_2000mm"] = n_extreme

            # All-zero would mean centroids fell outside MRMS CONUS grid
            all_zero = int(np.sum(values == 0))
            check["zero_count"] = all_zero

            is_bad = (check["flag"] != "OK" or n_extreme > 0 or
                      all_zero == len(values))
            status = "FAIL" if is_bad else "PASS"

            results.append(AuditResult(
                audit_id="mode_C2", scenario=scenario, probe="crs",
                status=status, detail=check,
                min_support=min_support, timestamp=ts,
            ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_C2", scenario=scenario, probe="crs",
            status="SKIP",
            detail={"note": "No CRS-sensitive features found for this scenario"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_c2_crs", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode C.2: Projection / CRS inconsistency detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
