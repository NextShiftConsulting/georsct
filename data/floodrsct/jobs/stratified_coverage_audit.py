#!/usr/bin/env python3
"""
stratified_coverage_audit.py -- Stage 5: Run all six coverage audits.

Orchestrates audit_a1 through audit_a6 for a scenario and produces
a consolidated report. Each audit runs independently and can also be
invoked standalone.

The six audits map to six decision geometries from the GeoRSCT benchmark:

  A1  Per-event support       -> Transfer probe (G4)
  A2  Coastal vs inland       -> Ranking probe (G2)
  A3  Levee-protected vs not  -> Ranking probe (G2)
  A4  County group sizes      -> Hierarchical probe (G5)
  A5  Adjacency coverage      -> Relational propagation probe (G6)
  A6  Outcome signal/stratum  -> All probes

Usage:
    python stratified_coverage_audit.py --scenario houston
    python stratified_coverage_audit.py --scenario houston --upload
    python stratified_coverage_audit.py --scenario all
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import SCENARIOS, get_s3_client, BUCKET

from audit_a1_event_support import audit as audit_a1
from audit_a2_coastal_inland import audit as audit_a2
from audit_a3_levee_protection import audit as audit_a3
from audit_a4_county_groups import audit as audit_a4
from audit_a5_adjacency_coverage import audit as audit_a5
from audit_a6_outcome_signal import audit as audit_a6

AUDITS = [
    ("A1_event_support", audit_a1),
    ("A2_coastal_inland", audit_a2),
    ("A3_levee_protection", audit_a3),
    ("A4_county_groups", audit_a4),
    ("A5_adjacency_coverage", audit_a5),
    ("A6_outcome_signal", audit_a6),
]


def run_scenario(scenario: str, min_support: int, upload: bool) -> dict:
    """Run all 6 audits for one scenario."""
    print(f"\n{'='*60}")
    print(f"  STAGE 5: STRATIFIED COVERAGE AUDIT -- {scenario}")
    print(f"{'='*60}\n")

    all_results = {}
    total_pass = 0
    total_fail = 0
    total_skip = 0

    for name, audit_fn in AUDITS:
        print(f"\n--- {name} ---")
        try:
            results = audit_fn(scenario, min_support, upload=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            results = []
            total_fail += 1

        for r in results:
            if r.status == "PASS":
                total_pass += 1
            elif r.status == "FAIL":
                total_fail += 1
            elif r.status == "SKIP":
                total_skip += 1

        all_results[name] = [r.__dict__ for r in results]

    # Summary
    total = total_pass + total_fail + total_skip
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {scenario}")
    print(f"  {total_pass}/{total} PASS, {total_fail} FAIL, {total_skip} SKIP")
    print(f"{'='*60}\n")

    report = {
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "min_support": min_support,
        "summary": {
            "total_checks": total,
            "pass": total_pass,
            "fail": total_fail,
            "skip": total_skip,
        },
        "audits": all_results,
    }

    if upload:
        s3 = get_s3_client()
        key = f"evidence/qa/coverage_audit_{scenario}.json"
        s3.put_object(
            Bucket=BUCKET, Key=key,
            Body=json.dumps(report, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        print(f"Uploaded s3://{BUCKET}/{key}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Stratified coverage audit (all 6 checks)"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=SCENARIOS + ["all"],
    )
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        run_scenario(scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
