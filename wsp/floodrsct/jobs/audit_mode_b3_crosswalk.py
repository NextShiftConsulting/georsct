#!/usr/bin/env python3
"""
audit_mode_b3_crosswalk.py -- GeoRSCT Mode B.3: Crosswalk Gap.

Checks join hit rates for features that depend on spatial crosswalks.
A low hit rate means the crosswalk is dropping data -- ZCTAs that
should have values are getting nulls because the join key doesn't match.

Crosswalk-dependent features:
  - NFIP claims: reportedZipcode -> ZCTA (ZIP != ZCTA for ~5% of records)
  - 311 reports: geocoded lat/lon -> ZCTA (geocoding accuracy varies)
  - HWM: spatial join nearest ZCTA centroid (distance-dependent)
  - Station assignment: tide gauge -> broadcast to all scenario ZCTAs

For each feature, computes:
  - join_rate = n_nonnull / n_expected (where n_expected = all ZCTAs in scenario)
  - For NFIP: also checks whether any disaster declaration covers this scenario

PASS if join_rate > 0.1 for features expected to have broad coverage (NFIP).
FAIL if a feature that should have coverage shows < 10% hit rate.

Usage:
    python audit_mode_b3_crosswalk.py --scenario houston
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

# Features that depend on crosswalk joins, with expected coverage
CROSSWALK_FEATURES = {
    "nfip_event_claims": {
        "join_type": "ZIP -> ZCTA",
        "expected_coverage": 0.10,  # at least 10% of ZCTAs should have claims
        "scenarios": ["houston", "new_orleans", "nyc",
                      "southwest_florida", "riverside_coachella"],
    },
    "flood_311_count": {
        "join_type": "geocoded lat/lon -> ZCTA",
        "expected_coverage": 0.05,
        "scenarios": ["houston", "nyc"],
    },
    "hwm_max_ft": {
        "join_type": "spatial nearest ZCTA centroid",
        "expected_coverage": 0.01,  # HWMs are sparse by nature
        "scenarios": ["houston", "southwest_florida", "nyc",
                      "riverside_coachella"],
    },
    "tidal_surge_max_m": {
        "join_type": "station broadcast to all ZCTAs",
        "expected_coverage": 0.90,  # broadcast should cover all ZCTAs
        "scenarios": ["houston", "new_orleans", "southwest_florida", "nyc"],
    },
}


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    n_total_rows = len(df)
    n_zctas = df["zcta_id"].nunique() if "zcta_id" in df.columns else 0

    for col, spec in CROSSWALK_FEATURES.items():
        if scenario not in spec["scenarios"]:
            continue
        if col not in df.columns:
            results.append(AuditResult(
                audit_id="mode_B3", scenario=scenario, mode="crosswalk",
                status="FAIL",
                detail={
                    "column": col,
                    "join_type": spec["join_type"],
                    "error": "Column missing from processed parquet",
                },
                min_support=min_support, timestamp=ts,
            ))
            continue

        n_nonnull = int(df[col].notna().sum())
        n_positive = int((df[col].fillna(0) > 0).sum()) if col != "tidal_surge_max_m" else n_nonnull
        join_rate = n_nonnull / max(n_total_rows, 1)
        positive_rate = n_positive / max(n_total_rows, 1)

        # For count features, also check how many are > 0
        # (nonnull but zero is a valid join result for NFIP)
        threshold = spec["expected_coverage"]
        status = "PASS" if join_rate >= threshold else "FAIL"

        detail = {
            "column": col,
            "join_type": spec["join_type"],
            "total_rows": n_total_rows,
            "total_zctas": n_zctas,
            "nonnull": n_nonnull,
            "join_rate": round(join_rate, 4),
            "expected_min": threshold,
        }

        if col not in ("tidal_surge_max_m",):
            detail["positive_count"] = n_positive
            detail["positive_rate"] = round(positive_rate, 4)

        # Extra: for broadcast features, check unique value count
        if spec["join_type"] == "station broadcast to all ZCTAs":
            n_unique = int(df[col].dropna().nunique())
            detail["unique_values"] = n_unique
            if n_unique == 1:
                detail["warning"] = "Single value broadcast -- see Mode B.2"

        results.append(AuditResult(
            audit_id="mode_B3", scenario=scenario, mode="crosswalk",
            status=status, detail=detail,
            min_support=min_support, timestamp=ts,
        ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_B3", scenario=scenario, mode="crosswalk",
            status="SKIP",
            detail={"note": "No crosswalk-dependent features for this scenario"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_b3_crosswalk", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode B.3: Crosswalk gap detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
