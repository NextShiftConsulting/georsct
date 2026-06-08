#!/usr/bin/env python3
"""
audit_a1_event_support.py -- Stage 5 Audit A1: Per-event support.

Checks whether each event in a scenario has enough populated rows to
support the transfer probe (G4). The transfer probe trains on one event
and evaluates on another; each event must be independently viable.

For each event, counts:
  - Total rows (zcta_id x event)
  - Non-null rainfall (core feature)
  - Non-null outcome signals (hwm_max_ft, nfip_event_claims)

PASS if every event has >= min_support rows with non-null core features.
FAIL if any event falls below the threshold.

Usage:
    python audit_a1_event_support.py --scenario houston
    python audit_a1_event_support.py --scenario houston --min-support 20
    python audit_a1_event_support.py --scenario houston --upload
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

CORE_FEATURES = ["rainfall_total_mm", "total_rainfall_mm"]
OUTCOME_FEATURES = ["hwm_max_ft", "nfip_event_claims", "flood_311_count"]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    df = load_processed_parquet(s3, scenario)

    if "event" not in df.columns:
        return [AuditResult(
            audit_id="P1", scenario=scenario, mode="transfer",
            status="FAIL",
            detail={"error": "No 'event' column in processed parquet"},
            min_support=min_support,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )]

    events = sorted(df["event"].unique())
    results = []

    for event in events:
        edf = df[df["event"] == event]
        total = len(edf)

        # Find whichever rainfall column exists
        rain_col = None
        for c in CORE_FEATURES:
            if c in edf.columns:
                rain_col = c
                break

        rain_nonnull = int(edf[rain_col].notna().sum()) if rain_col else 0

        # Outcome signals
        outcome_counts = {}
        for col in OUTCOME_FEATURES:
            if col in edf.columns:
                outcome_counts[col] = int(edf[col].notna().sum())

        viable = rain_nonnull >= min_support
        status = "PASS" if viable else "FAIL"

        results.append(AuditResult(
            audit_id="P1",
            scenario=scenario,
            mode="transfer",
            status=status,
            detail={
                "event": event,
                "total_rows": total,
                "rainfall_col": rain_col,
                "rainfall_nonnull": rain_nonnull,
                "outcome_signals": outcome_counts,
            },
            min_support=min_support,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    write_evidence(results, "a1_event_support", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A1: Per-event support")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
