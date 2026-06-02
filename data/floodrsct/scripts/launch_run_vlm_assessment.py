#!/usr/bin/env python3
"""launch_run_vlm_assessment.py -- Phase R4.3: VLM flood risk assessment.

Sends map image + text evidence to a VLM, collects structured risk
assessments per ZCTA. One job per (scenario, vlm) combination.
API calls only — no GPU needed.

Resource: ml.m5.large (2 vCPU, 8 GB). API-bound, not compute-bound.
Gemini rate-limited to 15 RPM so houston (~400 ZCTAs) takes ~30 min.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
VLMS = ["gemini", "nova", "qwen"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(
        f"vlm-{args.vlm}-{args.scenario.replace('_', '-')}"
    )

    launch_processing_job(
        job_name=job_name,
        job_script="run_vlm_assessment.py",
        job_args=["--scenario", args.scenario, "--vlm", args.vlm, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
