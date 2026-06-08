#!/usr/bin/env python3
"""
audit_a2_coastal_inland.py -- Stage 5 Audit A2: Coastal vs inland balance.

Checks whether surge/tide features have sufficient coverage in both
coastal and inland ZCTAs. The ranking probe (G2) uses Moran's I on
residuals; if all non-null surge values cluster in one zone, the
spatial statistic is underpowered.

Strata definition:
  - coastal_distance_m <= 20_000 m  -> "coastal"
  - coastal_distance_m >  20_000 m  -> "inland"

For scenarios without coastal_distance_m, uses tidal_surge_max_m
presence as a proxy: if tidal features exist for the scenario but
every ZCTA gets the same broadcast value, that's a different issue
(detected by strat_sampler_qa constant-column check, not here).

PASS if both strata have >= min_support rows with non-null surge.
SKIP for scenarios with no coastal/tidal features (riverside_coachella).

Usage:
    python audit_a2_coastal_inland.py --scenario southwest_florida
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

COASTAL_THRESHOLD_M = 20_000
SURGE_COLS = ["tidal_surge_max_m", "max_surge_m", "max_water_level_m"]
TIDAL_SCENARIOS = ["houston", "new_orleans", "southwest_florida", "nyc"]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()

    if scenario not in TIDAL_SCENARIOS:
        result = AuditResult(
            audit_id="P2", scenario=scenario, mode="ranking",
            status="SKIP",
            detail={"reason": "No tidal/coastal features for this scenario"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a2_coastal_inland", scenario, s3=s3, upload=upload)
        return [result]

    df = load_processed_parquet(s3, scenario)

    # Find surge column
    surge_col = None
    for c in SURGE_COLS:
        if c in df.columns:
            surge_col = c
            break

    if surge_col is None:
        result = AuditResult(
            audit_id="P2", scenario=scenario, mode="ranking",
            status="FAIL",
            detail={"error": "No surge column found", "checked": SURGE_COLS},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a2_coastal_inland", scenario, s3=s3, upload=upload)
        return [result]

    results = []

    if "coastal_distance_m" in df.columns:
        # Direct coastal/inland split
        coastal = df[df["coastal_distance_m"] <= COASTAL_THRESHOLD_M]
        inland = df[df["coastal_distance_m"] > COASTAL_THRESHOLD_M]

        for label, subset in [("coastal", coastal), ("inland", inland)]:
            n_total = len(subset)
            n_surge = int(subset[surge_col].notna().sum())
            status = "PASS" if n_surge >= min_support else "FAIL"

            results.append(AuditResult(
                audit_id="P2", scenario=scenario, mode="ranking",
                status=status,
                detail={
                    "stratum": label,
                    "threshold_m": COASTAL_THRESHOLD_M,
                    "total_rows": n_total,
                    "surge_col": surge_col,
                    "surge_nonnull": n_surge,
                },
                min_support=min_support, timestamp=ts,
            ))
    else:
        # No coastal_distance_m -- report surge coverage overall
        n_total = len(df)
        n_surge = int(df[surge_col].notna().sum())
        n_unique = int(df[surge_col].dropna().nunique())

        results.append(AuditResult(
            audit_id="P2", scenario=scenario, mode="ranking",
            status="PASS" if n_surge >= min_support else "FAIL",
            detail={
                "stratum": "all_zctas",
                "note": "No coastal_distance_m column; reporting overall surge coverage",
                "total_rows": n_total,
                "surge_col": surge_col,
                "surge_nonnull": n_surge,
                "surge_unique_values": n_unique,
            },
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "a2_coastal_inland", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A2: Coastal vs inland balance")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
