#!/usr/bin/env python3
"""
stratified_coverage_audit.py -- Stage 5: Run all 12 coverage audits.

Orchestrates probe-mapped audits (A1-A6) and GeoRSCT failure-mode
audits (mode_A1, B2, B3, C1, C3, D3) for a scenario and produces
a consolidated report. Each audit runs independently and can also be
invoked standalone.

Probe-mapped audits (stratum sufficiency):
  A1  Per-event support       -> Transfer probe (G4)
  A2  Coastal vs inland       -> Ranking probe (G2)
  A3  Levee-protected vs not  -> Ranking probe (G2)
  A4  County group sizes      -> Hierarchical probe (G5)
  A5  Adjacency coverage      -> Relational propagation probe (G6)
  A6  Outcome signal/stratum  -> All probes

GeoRSCT failure-mode audits (data quality):
  mode_A1  Autocorrelation leakage   -> random vs spatial split
  mode_B2  Scale mismatch            -> broadcast/coarse detection
  mode_B3  Crosswalk gap             -> join hit rates
  mode_C1  Vintage drift             -> feature vintage vs event year
  mode_C3  Spatial missingness bias  -> systematic null patterns
  mode_D3  Interp/extrap mismatch    -> cross-event distribution overlap

Usage:
    python stratified_coverage_audit.py --scenario houston
    python stratified_coverage_audit.py --scenario houston --upload
    python stratified_coverage_audit.py --scenario all
    python stratified_coverage_audit.py --scenario houston --probes-only
    python stratified_coverage_audit.py --scenario houston --modes-only
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
from audit_mode_a1_leakage import audit as audit_mode_a1
from audit_mode_b2_scale import audit as audit_mode_b2
from audit_mode_b3_crosswalk import audit as audit_mode_b3
from audit_mode_c1_vintage import audit as audit_mode_c1
from audit_mode_c3_missingness import audit as audit_mode_c3
from audit_mode_d3_transfer import audit as audit_mode_d3

PROBE_AUDITS = [
    ("A1_event_support", audit_a1),
    ("A2_coastal_inland", audit_a2),
    ("A3_levee_protection", audit_a3),
    ("A4_county_groups", audit_a4),
    ("A5_adjacency_coverage", audit_a5),
    ("A6_outcome_signal", audit_a6),
]

MODE_AUDITS = [
    ("mode_A1_leakage", audit_mode_a1),
    ("mode_B2_scale", audit_mode_b2),
    ("mode_B3_crosswalk", audit_mode_b3),
    ("mode_C1_vintage", audit_mode_c1),
    ("mode_C3_missingness", audit_mode_c3),
    ("mode_D3_transfer", audit_mode_d3),
]


def run_scenario(scenario: str, min_support: int, seed: int,
                 upload: bool,
                 run_probes: bool = True,
                 run_modes: bool = True) -> dict:
    """Run selected audits for one scenario."""
    print(f"\n{'='*60}")
    print(f"  STAGE 5: STRATIFIED COVERAGE AUDIT -- {scenario}")
    print(f"{'='*60}\n")

    all_results = {}
    total_pass = 0
    total_fail = 0
    total_skip = 0

    audit_list = []
    if run_probes:
        audit_list.extend(PROBE_AUDITS)
    if run_modes:
        audit_list.extend(MODE_AUDITS)

    for name, audit_fn in audit_list:
        print(f"\n--- {name} ---")
        try:
            # mode_A1 takes an extra seed parameter
            if name == "mode_A1_leakage":
                results = audit_fn(scenario, min_support, seed, upload=False)
            else:
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
        "seed": seed,
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
        description="Stage 5: Stratified coverage audit (12 checks)"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=SCENARIOS + ["all"],
    )
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--probes-only", action="store_true",
                        help="Run only probe-mapped audits (A1-A6)")
    parser.add_argument("--modes-only", action="store_true",
                        help="Run only GeoRSCT mode audits (A1,B2,B3,C1,C3,D3)")
    args = parser.parse_args()

    run_probes = not args.modes_only
    run_modes = not args.probes_only

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        run_scenario(scenario, args.min_support, args.seed, args.upload,
                     run_probes=run_probes, run_modes=run_modes)


if __name__ == "__main__":
    main()
