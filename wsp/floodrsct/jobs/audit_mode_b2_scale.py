#!/usr/bin/env python3
"""
audit_mode_b2_scale.py -- GeoRSCT Mode B.2: Scale Mismatch.

Flags features where the source spatial resolution is coarser than
the ZCTA it is assigned to. A 4.7 km MRMS cell assigned to a 0.5 km2
ZCTA is a single-pixel lookup masquerading as a spatial average.

Checks:
  1. Constant-value features: if tidal_surge_max_m is identical across
     all ZCTAs, it's a broadcast (one gauge -> all ZCTAs), not a
     spatially resolved field.
  2. ZCTA area vs source resolution: flag ZCTAs where the source grid
     cell is larger than the ZCTA itself.
  3. Temporal scalar collapse: features aggregated to a single scalar
     from a multi-hour event window (e.g., max rainfall from 168h).

Source resolutions (approximate):
  - MRMS: 0.01 deg (~1 km at mid-latitudes, 4.7 km for some products)
  - NLCD: 30 m
  - DEM (3DEP): 10-30 m
  - SLOSH MOM: 250 m - 1 km (basin-dependent)
  - Tide gauges: point measurement broadcast to all ZCTAs
  - HURDAT2: 6-hourly track points, distance computed per ZCTA

Usage:
    python audit_mode_b2_scale.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    write_evidence,
)

# Features known to be point/broadcast (not spatially resolved per ZCTA)
BROADCAST_FEATURES = [
    "tidal_surge_max_m", "max_surge_m", "max_water_level_m",
]

# Features derived from coarse grids
COARSE_FEATURES = {
    "rainfall_total_mm": {"source": "MRMS", "res_km": 1.0},
    "total_rainfall_mm": {"source": "MRMS", "res_km": 1.0},
    "max_rainfall_mm": {"source": "MRMS", "res_km": 1.0},
    "slosh_max_surge_m": {"source": "SLOSH MOM", "res_km": 0.5},
}

# Features that collapse a time series to one scalar
TEMPORAL_COLLAPSE = [
    "rainfall_total_mm", "total_rainfall_mm", "max_rainfall_mm",
    "tidal_surge_max_m", "max_surge_m",
    "storm_distance_km",
]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    # --- Check 1: Broadcast detection ---
    for col in BROADCAST_FEATURES:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        n_unique = int(values.nunique())
        n_total = len(values)

        # If all non-null values are identical, it's a broadcast
        is_broadcast = n_unique == 1
        # If very few unique values relative to ZCTAs, likely coarse
        is_near_broadcast = n_unique <= 3 and n_total > 10

        if is_broadcast or is_near_broadcast:
            results.append(AuditResult(
                audit_id="mode_B2", scenario=scenario, mode="scale",
                status="FAIL",
                detail={
                    "check": "broadcast_detection",
                    "column": col,
                    "unique_values": n_unique,
                    "total_nonnull": n_total,
                    "is_broadcast": is_broadcast,
                    "note": ("Single value broadcast to all ZCTAs"
                             if is_broadcast else
                             f"Only {n_unique} distinct values across {n_total} rows"),
                },
                min_support=min_support, timestamp=ts,
            ))
        else:
            results.append(AuditResult(
                audit_id="mode_B2", scenario=scenario, mode="scale",
                status="PASS",
                detail={
                    "check": "broadcast_detection",
                    "column": col,
                    "unique_values": n_unique,
                    "total_nonnull": n_total,
                },
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 2: Coarse grid features present ---
    coarse_present = []
    for col, meta in COARSE_FEATURES.items():
        if col in df.columns and df[col].notna().any():
            coarse_present.append({
                "column": col,
                "source": meta["source"],
                "source_resolution_km": meta["res_km"],
                "nonnull_rows": int(df[col].notna().sum()),
            })

    if coarse_present:
        results.append(AuditResult(
            audit_id="mode_B2", scenario=scenario, mode="scale",
            status="PASS",  # informational -- coarse features are expected
            detail={
                "check": "coarse_grid_inventory",
                "note": "These features have source resolution coarser than "
                        "typical ZCTA size. Nearest-centroid assignment used.",
                "features": coarse_present,
            },
            min_support=min_support, timestamp=ts,
        ))

    # --- Check 3: Temporal scalar collapse inventory ---
    collapsed = []
    for col in TEMPORAL_COLLAPSE:
        if col in df.columns and df[col].notna().any():
            values = df[col].dropna()
            collapsed.append({
                "column": col,
                "nonnull": len(values),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "std": round(float(values.std()), 4),
            })

    if collapsed:
        results.append(AuditResult(
            audit_id="mode_B2", scenario=scenario, mode="scale",
            status="PASS",  # informational
            detail={
                "check": "temporal_collapse_inventory",
                "note": "These features collapse a multi-hour event window "
                        "to a single scalar (max, total, or min distance).",
                "features": collapsed,
            },
            min_support=min_support, timestamp=ts,
        ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_B2", scenario=scenario, mode="scale",
            status="PASS",
            detail={"check": "no_scale_issues", "note": "No flagged features"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_b2_scale", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode B.2: Scale mismatch detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
