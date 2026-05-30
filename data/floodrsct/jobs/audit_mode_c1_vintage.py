#!/usr/bin/env python3
"""
audit_mode_c1_vintage.py -- GeoRSCT Mode C.1: Vintage Drift.

Checks whether static/slow-drift features are temporally appropriate
for the events they are joined to. ACS 2022 demographic data joined
to a 2017 hurricane introduces up to 5 years of demographic drift.

For each (feature, event) pair, computes vintage_gap = event_year - feature_vintage.
Flags gaps > 3 years as WARN, > 5 years as FAIL.

Feature vintages (hardcoded from FEATURE_CONTRACT.yaml):
  - ACS demographics: 2022 5-year estimates (covers 2018-2022)
  - NLCD impervious: 2021
  - USACE levees: 2023 (inspection date varies)
  - USGS subsidence: 2020 (InSAR epoch)
  - MTBS burn scars: 2015-2023 composite
  - geocertdb2026 base features: 2024 compilation

Usage:
    python audit_mode_c1_vintage.py --scenario houston
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

# Feature vintage registry: column_prefix -> vintage year
FEATURE_VINTAGES = {
    "acs_": {"vintage": 2022, "source": "ACS 5-Year Estimates",
             "note": "Covers 2018-2022 pooled"},
    "svi_": {"vintage": 2022, "source": "CDC SVI",
             "note": "Based on ACS 2018-2022"},
    "flood_pct_": {"vintage": 2024, "source": "FEMA NFHL",
                   "note": "GeocertDB2026 compilation"},
    "impervious_pct": {"vintage": 2021, "source": "NLCD",
                       "note": "NLCD 2021 release"},
    "levee_condition_rating": {"vintage": 2023, "source": "USACE NLD",
                               "note": "Inspection year varies"},
    "subsidence_rate_mm_yr": {"vintage": 2020, "source": "USGS InSAR",
                              "note": "InSAR epoch ~2016-2020"},
    "burn_scar_overlap": {"vintage": 2023, "source": "MTBS",
                          "note": "Composite 2015-2023"},
}

# Event years
EVENT_YEARS = {
    "harvey2017": 2017,
    "imelda2019": 2019,
    "beryl2024": 2024,
    "ida2021_nola": 2021,
    "ida2021_nyc": 2021,
    "henri2021": 2021,
    "ian2022": 2022,
    "helene2024": 2024,
    "milton2024": 2024,
    "hilary2023": 2023,
    "ar_flood_2023": 2023,
}

WARN_GAP = 3
FAIL_GAP = 5


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    if "event" not in df.columns:
        result = AuditResult(
            audit_id="mode_C1", scenario=scenario, probe="vintage",
            status="FAIL",
            detail={"error": "No 'event' column in processed parquet"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "mode_c1_vintage", scenario, s3=s3, upload=upload)
        return [result]

    events = sorted(df["event"].unique())
    all_cols = df.columns.tolist()

    for event in events:
        event_year = EVENT_YEARS.get(event)
        if event_year is None:
            continue

        for prefix, meta in FEATURE_VINTAGES.items():
            # Find columns matching this prefix
            matching = [c for c in all_cols
                        if c.startswith(prefix) or c == prefix]
            if not matching:
                continue

            # Check non-null in this event
            edf = df[df["event"] == event]
            present_cols = [c for c in matching if edf[c].notna().any()]
            if not present_cols:
                continue

            vintage = meta["vintage"]
            gap = abs(event_year - vintage)

            if gap > FAIL_GAP:
                status = "FAIL"
            elif gap > WARN_GAP:
                status = "FAIL"  # treat WARN as actionable
            else:
                status = "PASS"

            results.append(AuditResult(
                audit_id="mode_C1", scenario=scenario, probe="vintage",
                status=status,
                detail={
                    "event": event,
                    "event_year": event_year,
                    "feature_prefix": prefix,
                    "feature_vintage": vintage,
                    "source": meta["source"],
                    "vintage_gap_years": gap,
                    "warn_threshold": WARN_GAP,
                    "fail_threshold": FAIL_GAP,
                    "matching_columns": len(matching),
                    "populated_columns": len(present_cols),
                    "note": meta["note"],
                },
                min_support=min_support, timestamp=ts,
            ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_C1", scenario=scenario, probe="vintage",
            status="PASS",
            detail={"note": "No vintage-flagged features found in this scenario"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_c1_vintage", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode C.1: Vintage drift detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
