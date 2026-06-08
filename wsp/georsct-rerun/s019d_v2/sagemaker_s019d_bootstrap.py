#!/usr/bin/env python3
"""
SageMaker Launcher: S019D Bootstrap Confidence Intervals

Launches a lightweight processing job that:
  - Downloads seed_42/123/456 results JSONs from S3 (via IAM role in container)
  - Runs fold-level block bootstrap (B=1000) on N-ceiling per task
  - Bootstraps within-task kappa rho CIs
  - Outputs s019d_bootstrap_cis.json, s019d_bootstrap_cis.csv,
    s019d_bootstrap_summary.json

Instance: ml.m5.xlarge (4 vCPU, 16 GB) -- lightweight, ~5 min runtime
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from preflight import (
    preflight_check,
    S3Artifact,
    ArtifactGroup,
    BUCKET,
    WHEEL_PREFIX,
)

CODE_PREFIX = "rsct_code/series_019_v2/s019d"
OUTPUT_PREFIX = "rsct_curriculum/series_019_v2/results/s019d_bootstrap"
SEEDS = [42, 123, 456]


def get_image_uri(region="us-east-1"):
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v1.9"
    )


def bootstrap_code_group() -> ArtifactGroup:
    return ArtifactGroup("bootstrap code", [
        S3Artifact(BUCKET, f"{CODE_PREFIX}/run_s019d_bootstrap.py"),
        S3Artifact(BUCKET, f"{CODE_PREFIX}/bootstrap_s019d.sh"),
    ])


def bootstrap_results_group() -> ArtifactGroup:
    """Verify all three seed results exist before launching."""
    artifacts = []
    for seed in SEEDS:
        artifacts.append(
            S3Artifact(BUCKET,
                       f"rsct_curriculum/series_019_v2/results/s019d/seed_{seed}/s019d_results.json",
                       description=f"S019D seed_{seed} results")
        )
    return ArtifactGroup("s019d results (all seeds)", artifacts)


def main():
    parser = argparse.ArgumentParser(
        description="Launch S019D Bootstrap CI job on SageMaker"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Pre-flight check only -- do not launch job")
    parser.add_argument("--instance-type", default="ml.m5.xlarge",
                        help="SageMaker instance type (default: ml.m5.xlarge)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip S3 artifact pre-flight")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap replicates (default: 1000)")
    args = parser.parse_args()

    region = "us-east-1"
    session = boto3.Session(profile_name="nsc-swarm", region_name=region)
    aws_account_id = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{aws_account_id}:role/SageMakerExecutionRole"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"s019d-bootstrap-ci-{timestamp}"
    image_uri = get_image_uri(region)

    print("=" * 60)
    print("S019D Bootstrap CI")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"B:        {args.n_boot}")
    print(f"Seeds:    {SEEDS}")
    print(f"Method:   fold-level block bootstrap (5-fold resample)")
    print(f"Output:   s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print("=" * 60)

    if not args.skip_preflight:
        artifacts = [
            bootstrap_code_group(),
            bootstrap_results_group(),
        ]
        ok = preflight_check(artifacts, region=region, dry_run=args.dry_run)
        if not ok:
            print("\nFix missing artifacts before launching. Job NOT started.")
            sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch: {job_name}")
        print(f"[DRY RUN] Output -> s3://{BUCKET}/{OUTPUT_PREFIX}/")
        return

    container_args = [
        "--output-dir", "/opt/ml/processing/output",
        "--n-boot", str(args.n_boot),
        "--seeds", "42", "123", "456",
    ]

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 10,
            }
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/code/bootstrap_s019d.sh",
            ],
            "ContainerArguments": container_args,
        },
        "RoleArn": role_arn,
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{CODE_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 1800},  # 30 min safety
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        },
    }

    sm = session.client("sagemaker")
    response = sm.create_processing_job(**config)

    print(f"\nJob launched: {response['ProcessingJobArn']}")
    print(f"\nMonitor:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
          f"--processing-job-name {job_name} --region {region} --profile nsc-swarm")
    print(f"  MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs "
          f"--log-stream-name-prefix {job_name} --follow --region {region} --profile nsc-swarm")
    print(f"\nResults will be at:")
    print(f"  s3://{BUCKET}/{OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
