#!/usr/bin/env python3
"""
stratified_coverage_audit.py -- Stage 5: Run all 15 coverage audits.

Orchestrates two audit layers for a scenario and produces a consolidated
report. Each audit runs independently and can also be invoked standalone.

Layer 0: Dataset-support probes (is the substrate admissible?)
  P1  Per-event support       -> precondition for D.3 transfer
  P2  Coastal vs inland       -> precondition for A.2 heterogeneity
  P3  Levee-protected vs not  -> precondition for A.2 heterogeneity
  P4  County group sizes      -> precondition for A.1 blocked CV
  P5  Adjacency coverage      -> precondition for graph/spatial-lag
  P6  Outcome signal/stratum  -> precondition for C.3 and D.2

Layer 1: GeoRSCT mode audits (which failure modes are active?)
  A.1  Autocorrelation leakage   -> random vs spatial split
  A.2  Geographic heterogeneity  -> per-stratum CV divergence
  B.1  MAUP / partition drift    -> ZCTA boundary misalignment
  B.2  Scale mismatch            -> broadcast/coarse detection
  B.3  Crosswalk gap             -> join hit rates
  C.1  Vintage drift             -> feature vintage vs event year
  C.2  CRS inconsistency         -> projection/datum mismatch
  C.3  Spatial missingness bias  -> systematic null patterns
  D.3  Interp/extrap mismatch    -> cross-event distribution overlap

Usage:
    python stratified_coverage_audit.py --scenario houston
    python stratified_coverage_audit.py --scenario houston --upload
    python stratified_coverage_audit.py --scenario all
    python stratified_coverage_audit.py --scenario houston --support-only
    python stratified_coverage_audit.py --scenario houston --modes-only
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import SCENARIOS, get_s3_client, BUCKET

from support_p1_event_support import audit as support_p1
from support_p2_coastal_inland import audit as support_p2
from support_p3_levee_protection import audit as support_p3
from support_p4_county_groups import audit as support_p4
from support_p5_adjacency_coverage import audit as support_p5
from support_p6_outcome_signal import audit as support_p6
from audit_mode_a1_leakage import audit as audit_mode_a1
from audit_mode_a2_heterogeneity import audit as audit_mode_a2
from audit_mode_b1_maup import audit as audit_mode_b1
from audit_mode_b2_scale import audit as audit_mode_b2
from audit_mode_b3_crosswalk import audit as audit_mode_b3
from audit_mode_c1_vintage import audit as audit_mode_c1
from audit_mode_c2_crs import audit as audit_mode_c2
from audit_mode_c3_missingness import audit as audit_mode_c3
from audit_mode_d3_transfer import audit as audit_mode_d3

SUPPORT_PROBES = [
    ("P1_event_support", support_p1),
    ("P2_coastal_inland", support_p2),
    ("P3_levee_protection", support_p3),
    ("P4_county_groups", support_p4),
    ("P5_adjacency_coverage", support_p5),
    ("P6_outcome_signal", support_p6),
]

MODE_AUDITS = [
    ("mode_A1_leakage", audit_mode_a1),
    ("mode_A2_heterogeneity", audit_mode_a2),
    ("mode_B1_maup", audit_mode_b1),
    ("mode_B2_scale", audit_mode_b2),
    ("mode_B3_crosswalk", audit_mode_b3),
    ("mode_C1_vintage", audit_mode_c1),
    ("mode_C2_crs", audit_mode_c2),
    ("mode_C3_missingness", audit_mode_c3),
    ("mode_D3_transfer", audit_mode_d3),
]


def run_scenario(scenario: str, min_support: int, seed: int,
                 upload: bool,
                 run_support: bool = True,
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
    if run_support:
        audit_list.extend(SUPPORT_PROBES)
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
        description="Stage 5: Stratified coverage audit (15 checks)"
    )
    parser.add_argument(
        "--scenario", required=True,
        choices=SCENARIOS + ["all"],
    )
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--support-only", action="store_true",
                        help="Run only Layer 0 support probes (P1-P6)")
    parser.add_argument("--modes-only", action="store_true",
                        help="Run only Layer 1 GeoRSCT mode audits")
    args = parser.parse_args()

    run_support = not args.modes_only
    run_modes = not args.support_only

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        run_scenario(scenario, args.min_support, args.seed, args.upload,
                     run_support=run_support, run_modes=run_modes)


if __name__ == "__main__":
    main()
