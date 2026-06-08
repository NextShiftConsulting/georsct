#!/usr/bin/env python3
"""
audit_mode_d2_flattening.py -- GeoRSCT Mode D.2: Ceiling-Induced Architecture Flattening.

NOT_READY: Requires R0 vs R1 vs R2 performance comparisons.

Detects whether all solvers converge to similar performance because
the representation ceiling -- not solver capacity -- is the binding
constraint. If Ridge, HistGBDT, and XGBoost all score ~0.65 on R0,
adding a GNN won't help until the representation improves.

Will check:
  1. Cross-solver performance spread on same representation
  2. Representation uplift: R1-R0 and R2-R1 deltas
  3. Whether audit-flagged failure modes (B.1, C.1) predict
     which representation fix helps most

Requires: predictions from multiple solvers on R0, R1, R2.

Usage:
    python audit_mode_d2_flattening.py --scenario houston
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
        audit_id="mode_D2", mode="flattening", scenario=scenario,
        status="NOT_READY",
        detail={
            "note": (
                "Architecture Flattening detection requires cross-solver "
                "comparisons on R0/R1/R2 representations. Train Ridge + "
                "HistGBDT + XGBoost on R0 first, then compare spread."
            ),
            "requires": [
                "R0 predictions from Ridge, HistGBDT, XGBoost",
                "R1 predictions from same solvers",
                "R2 predictions from same solvers",
            ],
            "checks_planned": [
                "cross-solver performance spread per representation",
                "representation uplift (R1-R0, R2-R1) by solver",
                "audit-flag -> uplift correlation (B.1/MAUP, C.1/vintage)",
            ],
        },
        min_support=min_support, timestamp=ts,
        recommendation="Train R0 with Ridge + HistGBDT + XGBoost, then re-run.",
    )
    write_evidence([result], "mode_d2_flattening", scenario,
                   s3=get_s3_client(), upload=upload)
    return [result]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode D.2: Architecture flattening detection (NOT_READY)"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
