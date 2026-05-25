#!/usr/bin/env python3
"""
SageMaker Launcher: Flood Overlay-Only (Lane 3)

Reads pre-fetched county JSONs from S3, does spatial overlay, outputs
flood_zones_zcta.parquet. No FEMA API calls.

Instance: ml.m5.2xlarge (8 vCPU, 32 GB — overlay is memory-bound)
Runtime:  ~2-4 hours (overlay only, no fetch wait)
Cost:     ~$1-2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_overlay"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_overlay_only.py",
    "entrypoint_flood_overlay.sh",
]


def get_image_uri(region="us-east-1"):
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.5-cpu-py311"
    )


def deploy_code(s3):
    for fname in CODE_FILES:
        local = CODE_DIR / fname
        if not local.exists():
            print(f"  ERROR: {local} not found")
            sys.exit(1)
        key = f"{CODE_PREFIX}/{fname}"
        s3.upload_file(str(local), BUCKET, key)
        print(f"  Uploaded: s3://{BUCKET}/{key}")


def check_fetch_files(s3):
    """Count available fetch files on S3."""
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="rsct_curriculum/series_018/processed/flood_fetch/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "summary" not in obj["Key"]:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Launch flood overlay-only SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--force", action="store_true", help="Launch even if <2000 fetch files")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    # Pre-flight: check how many fetch files exist
    print("\n=== PRE-FLIGHT CHECK ===")
    n_fetch = check_fetch_files(s3)
    print(f"  Fetch files on S3: {n_fetch}/~3090")

    if n_fetch < 2000 and not args.force:
        print(f"\n  WARNING: Only {n_fetch} fetch files available.")
        print("  Wait for lane 2 (fetch-only) to complete, or use --force to launch anyway.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-flood-overlay-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Overlay-Only (Lane 3)")
    print("=" * 60)
    print(f"Job:       {job_name}")
    print(f"Instance:  {args.instance_type}")
    print(f"Fetch:     {n_fetch} counties on S3")
    print(f"Image:     {image_uri}")
    print(f"Output:    s3://{BUCKET}/{DATA_PREFIX}/flood_zones_zcta.parquet")
    print("=" * 60)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch with above config.")
        return

    sm = boto3.client("sagemaker", **_aws)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 50,
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash",
                "/opt/ml/processing/input/code/entrypoint_flood_overlay.sh",
            ],
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
            {
                "InputName": "crosswalk",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "tiger",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{TIGER_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/tiger",
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
                        "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 21600},  # 6 hours
        "Environment": {
            "PYTHONUNBUFFERED": "1",
        },
    }

    response = sm.create_processing_job(**config)

    print(f"\nJob launched: {response['ProcessingJobArn']}")
    print(f"\nMonitor:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
          f"--processing-job-name {job_name} --region {REGION} --profile nsc-swarm "
          f"--query 'ProcessingJobStatus'")
    print(f"\nLive logs:")
    print(f"  MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs "
          f"--log-stream-name-prefix {job_name} --follow "
          f"--profile nsc-swarm --region {REGION}")


if __name__ == "__main__":
    main()
