#!/usr/bin/env python3
"""
SageMaker Launcher: FEMA NFHL Fetch-Only (Lane 2)

Pure download job — 16 threads, no overlay, dumps county JSONs to S3.
Skips counties already on S3. Companion to run_flood_zones.py which
reads the cached fetch files on startup.

Instance: ml.m5.large (2 vCPU, 8 GB — fetch is I/O-bound, not memory)
Runtime:  ~6-10 hours estimated (FEMA API is the bottleneck)
Cost:     ~$0.60-1.00
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_fetch"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"
OUTPUT_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_fetch_only.py",
    "entrypoint_flood_fetch.sh",
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


def main():
    parser = argparse.ArgumentParser(description="Launch flood fetch-only SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-flood-fetch-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Fetch-Only (Lane 2)")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Threads:  {args.threads}")
    print(f"Image:    {image_uri}")
    print(f"Output:   s3://{BUCKET}/{DATA_PREFIX}/flood_fetch/")
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
                "VolumeSizeInGB": 30,
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash",
                "/opt/ml/processing/input/code/entrypoint_flood_fetch.sh",
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
                        "S3Uri": f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 43200},  # 12 hours max
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
