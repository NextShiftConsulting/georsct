#!/usr/bin/env python3
"""
audit_mode_a3_smooth_map.py -- GeoRSCT Mode A.3: Smooth-Map Illusion.

NOT_READY: Requires predicted surfaces from R0/R1/R2 models.

Detects whether spatial interpolation between training points creates
artificially smooth prediction surfaces that mask local discontinuities
(e.g., levee boundaries, drainage divides, land-use transitions).

Will check:
  1. Predicted surface spatial autocorrelation (Moran's I on residuals)
  2. Discontinuity preservation at known boundaries (levee, HUC)
  3. Variance ratio: predicted vs observed at boundary ZCTAs

Requires: model predictions per ZCTA, not just assembled features.

Usage:
    python audit_mode_a3_smooth_map.py --scenario houston
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
        audit_id="mode_A3", mode="smooth_map", scenario=scenario,
        status="NOT_READY",
        detail={
            "note": (
                "Smooth-Map Illusion detection requires predicted surfaces "
                "from R0/R1/R2 models. Run model training first, then re-run "
                "this audit with --predictions-path."
            ),
            "requires": ["model predictions per ZCTA-event"],
            "checks_planned": [
                "Moran's I on prediction residuals",
                "discontinuity preservation at levee/HUC boundaries",
                "variance ratio: predicted vs observed at boundary ZCTAs",
            ],
        },
        min_support=min_support, timestamp=ts,
        recommendation="Train R0 baseline, then re-run with predictions.",
    )
    write_evidence([result], "mode_a3_smooth_map", scenario,
                   s3=get_s3_client(), upload=upload)
    return [result]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode A.3: Smooth-map illusion detection (NOT_READY)"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
