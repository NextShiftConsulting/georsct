#!/usr/bin/env python3
"""launch_score_vlm_quality.py -- Phase R4.4: Launch evidence-grounding audit.

Runs score_vlm_quality.py for a (scenario, vlm) pair as a SageMaker
Processing job. API calls only (for evidence text loading), no GPU.

Resource: ml.m5.large (2 vCPU, 8 GB). Lightweight -- just parquet loads
and string matching. Each job takes ~5-10 min.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
VLMS = ["gpt4o", "gemini_flash", "jina", "nova", "qwen"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(
        f"quality-{args.vlm.replace('_', '-')}-{args.scenario.replace('_', '-')}"
    )

    job_args = ["--scenario", args.scenario, "--vlm", args.vlm, "--upload"]

    launch_processing_job(
        job_name=job_name,
        job_script="score_vlm_quality.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="",
        env_overrides={},
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
