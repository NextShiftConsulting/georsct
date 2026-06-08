#!/usr/bin/env python3
"""
audit_mode_d1_ceiling_drift.py -- GeoRSCT Mode D.1: Ceiling-Aggregation Drift.

NOT_READY: Requires per-event accuracy breakdowns from R0/R1/R2 models.

Detects whether aggregated performance metrics hide per-event or
per-stratum degradation. A model that scores R2=0.70 overall may
score 0.90 on Harvey but 0.30 on Beryl, with the aggregate hiding
the transfer failure.

Will check:
  1. Per-event performance spread (max - min across events)
  2. Per-county performance spread within a scenario
  3. Correlation between stratum size and stratum performance
     (small strata systematically worse = aggregation ceiling)

Requires: model predictions and per-fold/per-event metrics.

Usage:
    python audit_mode_d1_ceiling_drift.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, write_evidence, get_s3_client,
)


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    ts = datetime.now(timezone.utc).isoformat()
    result = AuditResult(
        audit_id="mode_D1", mode="ceiling_drift", scenario=scenario,
        status="NOT_READY",
        detail={
            "note": (
                "Ceiling-Aggregation Drift detection requires per-event "
                "accuracy breakdowns from trained models. Run R0 with "
                "leave-one-event-out evaluation first."
            ),
            "requires": ["per-event R2/MAE from R0 model"],
            "checks_planned": [
                "per-event performance spread (max - min)",
                "per-county performance spread",
                "correlation between stratum size and performance",
            ],
        },
        min_support=min_support, timestamp=ts,
        recommendation="Train R0 with leave-one-event-out, then re-run.",
    )
    write_evidence([result], "mode_d1_ceiling_drift", scenario,
                   s3=get_s3_client(), upload=upload)
    return [result]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode D.1: Ceiling-aggregation drift detection (NOT_READY)"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
