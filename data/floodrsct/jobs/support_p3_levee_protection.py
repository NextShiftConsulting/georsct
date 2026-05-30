#!/usr/bin/env python3
"""
audit_a3_levee_protection.py -- Stage 5 Audit A3: Levee-protected vs unprotected.

Checks whether the ranking probe (G2) has enough data in both
levee-protected and unprotected strata. If all non-null outcomes
sit behind levees (or none do), the probe cannot distinguish
infrastructure-mediated vs unmediated flood outcomes.

Strata definition:
  - levee_condition_rating IS NOT NULL -> "protected"
  - levee_condition_rating IS NULL     -> "unprotected"

Only applicable to scenarios with levee features: new_orleans, nyc.

Usage:
    python audit_a3_levee_protection.py --scenario new_orleans
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

LEVEE_SCENARIOS = ["new_orleans", "nyc"]
OUTCOME_COLS = ["hwm_max_ft", "nfip_event_claims", "flood_311_count"]


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()

    if scenario not in LEVEE_SCENARIOS:
        result = AuditResult(
            audit_id="P3", scenario=scenario, mode="ranking",
            status="SKIP",
            detail={"reason": f"No levee features for {scenario}"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a3_levee_protection", scenario, s3=s3, upload=upload)
        return [result]

    df = load_processed_parquet(s3, scenario)

    if "levee_condition_rating" not in df.columns:
        result = AuditResult(
            audit_id="P3", scenario=scenario, mode="ranking",
            status="FAIL",
            detail={"error": "levee_condition_rating column missing from parquet"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a3_levee_protection", scenario, s3=s3, upload=upload)
        return [result]

    protected = df[df["levee_condition_rating"].notna()]
    unprotected = df[df["levee_condition_rating"].isna()]

    results = []
    for label, subset in [("protected", protected), ("unprotected", unprotected)]:
        n_total = len(subset)

        # Check outcome coverage in this stratum
        outcome_counts = {}
        for col in OUTCOME_COLS:
            if col in subset.columns:
                outcome_counts[col] = int(subset[col].notna().sum())

        any_outcome = sum(outcome_counts.values())
        status = "PASS" if any_outcome >= min_support else "FAIL"

        results.append(AuditResult(
            audit_id="P3", scenario=scenario, mode="ranking",
            status=status,
            detail={
                "stratum": label,
                "total_rows": n_total,
                "outcome_nonnull": outcome_counts,
                "total_outcome_nonnull": any_outcome,
            },
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "a3_levee_protection", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A3: Levee protection balance")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
