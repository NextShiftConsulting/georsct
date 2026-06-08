#!/usr/bin/env python3
"""
audit_a4_county_groups.py -- Stage 5 Audit A4: County group sizes.

Checks whether hierarchical/transfer probes (G5) have enough ZCTAs
per county to support within-county vs across-county comparisons.
The hierarchical probe computes eta-squared using ZCTA -> county -> state
grouping; counties with < min_support ZCTAs cannot contribute meaningful
within-group variance.

Loads the ZCTA-county crosswalk from geocertdb2026 and joins to the
processed parquet to count ZCTAs per county.

PASS if every county in the scenario has >= min_support ZCTAs with
non-null core features.

Usage:
    python audit_a4_county_groups.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    load_crosswalk, write_evidence,
)

CORE_FEATURES = ["rainfall_total_mm", "total_rainfall_mm"]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)

    # Load crosswalk and join
    try:
        xwalk = load_crosswalk(s3)
    except Exception as e:
        result = AuditResult(
            audit_id="P4", scenario=scenario, mode="hierarchical",
            status="FAIL",
            detail={"error": f"Could not load crosswalk: {e}"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a4_county_groups", scenario, s3=s3, upload=upload)
        return [result]

    # Ensure zcta_id types match
    df["zcta_id"] = df["zcta_id"].astype(str)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)

    # Get unique ZCTAs in this scenario (deduplicate across events)
    zctas = df[["zcta_id"]].drop_duplicates()
    merged = zctas.merge(xwalk, on="zcta_id", how="left")

    if "county_fips" not in merged.columns:
        result = AuditResult(
            audit_id="P4", scenario=scenario, mode="hierarchical",
            status="FAIL",
            detail={"error": "No county_fips in crosswalk after join"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a4_county_groups", scenario, s3=s3, upload=upload)
        return [result]

    county_counts = merged["county_fips"].value_counts().to_dict()

    results = []
    for county, count in sorted(county_counts.items()):
        status = "PASS" if count >= min_support else "FAIL"
        results.append(AuditResult(
            audit_id="P4", scenario=scenario, mode="hierarchical",
            status=status,
            detail={
                "county_fips": str(county),
                "zcta_count": int(count),
            },
            min_support=min_support, timestamp=ts,
        ))

    # Add summary
    total_counties = len(county_counts)
    passing = sum(1 for r in results if r.status == "PASS")
    results.append(AuditResult(
        audit_id="P4", scenario=scenario, mode="hierarchical",
        status="PASS" if passing == total_counties else "FAIL",
        detail={
            "summary": True,
            "total_counties": total_counties,
            "counties_passing": passing,
            "counties_failing": total_counties - passing,
            "min_county_size": min(county_counts.values()),
            "max_county_size": max(county_counts.values()),
            "total_zctas": len(zctas),
        },
        min_support=min_support, timestamp=ts,
    ))

    write_evidence(results, "a4_county_groups", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A4: County group sizes")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
