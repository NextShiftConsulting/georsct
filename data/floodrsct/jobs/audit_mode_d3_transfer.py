#!/usr/bin/env python3
"""
audit_mode_d3_transfer.py -- GeoRSCT Mode D.3: Interp/Extrap Mismatch.

Checks whether feature distributions overlap across events within a
scenario. A model trained on harvey2017 rainfall distribution may
extrapolate poorly to beryl2024 if the distributions don't overlap.

For each numeric feature present in multiple events, computes:
  - Per-event mean, std, min, max
  - Overlap coefficient: fraction of one event's range covered by another
  - Distribution gap flag: events with non-overlapping interquartile ranges

PASS if all features have overlapping IQRs across events.
FAIL if any feature has events with non-overlapping IQRs.

Usage:
    python audit_mode_d3_transfer.py --scenario houston
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

# Features to check for distribution overlap
TRANSFER_FEATURES = [
    "rainfall_total_mm", "total_rainfall_mm", "max_rainfall_mm",
    "tidal_surge_max_m", "max_surge_m", "max_water_level_m",
    "storm_distance_km",
    "hwm_max_ft",
    "nfip_event_claims",
]


def _iqr_overlap(q1_a, q3_a, q1_b, q3_b) -> float:
    """Compute IQR overlap fraction. 0 = no overlap, 1 = complete overlap."""
    overlap_lo = max(q1_a, q1_b)
    overlap_hi = min(q3_a, q3_b)

    if overlap_hi <= overlap_lo:
        return 0.0

    overlap_width = overlap_hi - overlap_lo
    min_iqr = min(q3_a - q1_a, q3_b - q1_b)

    if min_iqr <= 0:
        return 0.0

    return min(1.0, overlap_width / min_iqr)


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    if "event" not in df.columns:
        result = AuditResult(
            audit_id="mode_D3", scenario=scenario, mode="transfer",
            status="FAIL",
            detail={"error": "No 'event' column"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "mode_d3_transfer", scenario, s3=s3, upload=upload)
        return [result]

    events = sorted(df["event"].unique())
    if len(events) < 2:
        result = AuditResult(
            audit_id="mode_D3", scenario=scenario, mode="transfer",
            status="SKIP",
            detail={"note": f"Only {len(events)} event(s); transfer needs >= 2",
                    "events": events},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "mode_d3_transfer", scenario, s3=s3, upload=upload)
        return [result]

    for col in TRANSFER_FEATURES:
        if col not in df.columns:
            continue
        if df[col].notna().sum() < min_support:
            continue

        # Compute per-event statistics
        event_stats = {}
        for event in events:
            values = df.loc[df["event"] == event, col].dropna().values
            if len(values) < 3:
                continue
            event_stats[event] = {
                "n": len(values),
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "q1": round(float(np.percentile(values, 25)), 4),
                "median": round(float(np.median(values)), 4),
                "q3": round(float(np.percentile(values, 75)), 4),
                "max": round(float(np.max(values)), 4),
            }

        if len(event_stats) < 2:
            continue

        # Pairwise IQR overlap
        event_list = sorted(event_stats.keys())
        min_overlap = 1.0
        worst_pair = None
        pair_details = []

        for i in range(len(event_list)):
            for j in range(i + 1, len(event_list)):
                e_a, e_b = event_list[i], event_list[j]
                s_a, s_b = event_stats[e_a], event_stats[e_b]

                overlap = _iqr_overlap(s_a["q1"], s_a["q3"],
                                       s_b["q1"], s_b["q3"])

                pair_details.append({
                    "event_a": e_a,
                    "event_b": e_b,
                    "iqr_overlap": round(overlap, 4),
                })

                if overlap < min_overlap:
                    min_overlap = overlap
                    worst_pair = (e_a, e_b)

        # FAIL if any pair has zero IQR overlap
        has_gap = min_overlap == 0.0
        status = "FAIL" if has_gap else "PASS"

        results.append(AuditResult(
            audit_id="mode_D3", scenario=scenario, mode="transfer",
            status=status,
            detail={
                "column": col,
                "n_events": len(event_stats),
                "per_event": event_stats,
                "pairwise_overlap": pair_details,
                "min_iqr_overlap": round(min_overlap, 4),
                "worst_pair": list(worst_pair) if worst_pair else None,
                "interpretation": (
                    f"IQR gap between {worst_pair[0]} and {worst_pair[1]} -- "
                    f"model trained on one may extrapolate on the other"
                    if has_gap else
                    "All event pairs have overlapping IQRs"
                ),
            },
            min_support=min_support, timestamp=ts,
        ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_D3", scenario=scenario, mode="transfer",
            status="PASS",
            detail={"note": "No transfer-relevant features found"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_d3_transfer", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode D.3: Interp/extrap mismatch detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
