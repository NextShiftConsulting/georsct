#!/usr/bin/env python3
"""
audit_a6_outcome_signal.py -- Stage 5 Audit A6: Outcome signal per stratum.

Checks whether outcome/label features (HWM, NFIP claims, 311 reports)
have sufficient non-null coverage across the strata that probes slice by.
All six probes ultimately need outcome signals to evaluate predictions;
if outcomes concentrate in one stratum, probes that cross strata are
underpowered.

Strata checked (where applicable):
  - by event (does each event have outcomes?)
  - by county (do outcomes span multiple counties?)
  - presence vs absence of infrastructure features

PASS if each applicable stratum has >= min_support rows with at least
one non-null outcome signal.

Usage:
    python audit_a6_outcome_signal.py --scenario houston
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

OUTCOME_COLS = ["hwm_max_ft", "nfip_event_claims", "flood_311_count"]


def _count_any_outcome(df, cols: list[str]) -> int:
    """Count rows where at least one outcome column is non-null."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return 0
    mask = df[present].notna().any(axis=1)
    return int(mask.sum())


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)

    available_outcomes = [c for c in OUTCOME_COLS if c in df.columns]
    if not available_outcomes:
        result = AuditResult(
            audit_id="P6", scenario=scenario, mode="all",
            status="FAIL",
            detail={"error": "No outcome columns found", "checked": OUTCOME_COLS},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a6_outcome_signal", scenario, s3=s3, upload=upload)
        return [result]

    results = []

    # --- Check 1: Overall outcome coverage ---
    total = len(df)
    any_outcome = _count_any_outcome(df, available_outcomes)
    per_col = {c: int(df[c].notna().sum()) for c in available_outcomes}

    results.append(AuditResult(
        audit_id="P6", scenario=scenario, mode="all",
        status="PASS" if any_outcome >= min_support else "FAIL",
        detail={
            "check": "overall",
            "total_rows": total,
            "any_outcome_nonnull": any_outcome,
            "per_column": per_col,
        },
        min_support=min_support, timestamp=ts,
    ))

    # --- Check 2: Per-event outcome coverage ---
    if "event" in df.columns:
        events = sorted(df["event"].unique())
        for event in events:
            edf = df[df["event"] == event]
            n_outcome = _count_any_outcome(edf, available_outcomes)
            e_per_col = {c: int(edf[c].notna().sum()) for c in available_outcomes}

            results.append(AuditResult(
                audit_id="P6", scenario=scenario, mode="all",
                status="PASS" if n_outcome >= min_support else "FAIL",
                detail={
                    "check": "per_event",
                    "event": event,
                    "total_rows": len(edf),
                    "any_outcome_nonnull": n_outcome,
                    "per_column": e_per_col,
                },
                min_support=min_support, timestamp=ts,
            ))

    # --- Check 3: Per-county outcome coverage ---
    try:
        xwalk = load_crosswalk(s3)
        df["zcta_id"] = df["zcta_id"].astype(str)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
        df_county = df.merge(xwalk, on="zcta_id", how="left")

        if "county_fips" in df_county.columns:
            counties = sorted(df_county["county_fips"].dropna().unique())
            for county in counties:
                cdf = df_county[df_county["county_fips"] == county]
                n_outcome = _count_any_outcome(cdf, available_outcomes)

                results.append(AuditResult(
                    audit_id="P6", scenario=scenario, mode="all",
                    status="PASS" if n_outcome >= min_support else "FAIL",
                    detail={
                        "check": "per_county",
                        "county_fips": str(county),
                        "total_rows": len(cdf),
                        "any_outcome_nonnull": n_outcome,
                    },
                    min_support=min_support, timestamp=ts,
                ))
    except Exception:
        pass  # Crosswalk unavailable; skip county check

    write_evidence(results, "a6_outcome_signal", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A6: Outcome signal per stratum")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
