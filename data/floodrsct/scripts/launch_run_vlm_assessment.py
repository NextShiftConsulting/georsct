#!/usr/bin/env python3
"""launch_run_vlm_assessment.py -- Phase R4.3: VLM flood risk assessment.

Sends map image + text evidence to a VLM, collects structured risk
assessments per ZCTA. One job per (scenario, vlm) combination.
API calls only — no GPU needed.

Resource: ml.m5.large (2 vCPU, 8 GB). API-bound, not compute-bound.
Gemini rate-limited to 15 RPM so houston (~400 ZCTAs) takes ~30 min.
"""

import argparse
import json
import sys
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
VLMS = ["gpt4o", "gemini", "jina", "nova", "qwen"]

# pip deps per VLM -- union installed for simplicity
# gpt4o/jina/qwen: openai (OpenAI-compatible endpoints)
# gemini: google-generativeai + Pillow (Image.open in adapter)
# nova: boto3 (already in base image via swarm_auth)
VLM_PIP = "openai google-generativeai Pillow"

# Secrets Manager name -> env var for each VLM that needs an API key.
# Nova uses Bedrock (IAM role), no key needed.
_VLM_SECRETS = {
    "gpt4o": ("swarmit/openai-api-key", "OPENAI_API_KEY"),
    "gemini": ("google-api-key", "GOOGLE_API_KEY"),
    "jina": ("jina-api-key", "JINA_API_KEY"),
    "qwen": ("openrouter-api-key", "OPENROUTER_API_KEY"),
}


def _fetch_vlm_env(vlm: str) -> dict[str, str]:
    """Fetch API key from Secrets Manager for the given VLM."""
    if vlm not in _VLM_SECRETS:
        return {}
    secret_name, env_var = _VLM_SECRETS[vlm]
    sm = boto3.client("secretsmanager", region_name="us-east-1", **get_aws_credentials())
    resp = sm.get_secret_value(SecretId=secret_name)
    secret = resp["SecretString"]
    # Some secrets are JSON {"api_key": "..."}, others are plain strings
    try:
        parsed = json.loads(secret)
        val = parsed.get("api_key") or parsed.get("key") or list(parsed.values())[0]
    except (json.JSONDecodeError, IndexError):
        val = secret
    return {env_var: val}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(
        f"vlm-{args.vlm}-{args.scenario.replace('_', '-')}"
    )

    env = _fetch_vlm_env(args.vlm) if not args.dry_run else {}

    launch_processing_job(
        job_name=job_name,
        job_script="run_vlm_assessment.py",
        job_args=["--scenario", args.scenario, "--vlm", args.vlm, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages=VLM_PIP,
        env_overrides=env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
