#!/usr/bin/env python3
"""
SageMaker Launcher: GeoParquet Assembly

Assembles the final geocert GeoParquet from all enrichment layers + TIGER/Line
ZCTA boundaries, runs Croissant validation gate, uploads to S3 release prefix.

Downloads TIGER/Line 2022 ZCTA boundaries (~800 MB) from Census Bureau during job.
All data parquets are downloaded from S3 by the build script (not mounted).

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM, $0.23/hr)
Runtime:  10-20 min estimated (TIGER download + join + validation)
Cost:     ~$0.08 estimated

Mounts:
  Code:   s3://swarm-yrsn-datasets/rsct_code/geocert_v24/geoparquet/

Output: uploaded directly to S3 release prefix by the build script.

Usage:
  python sagemaker_geoparquet.py --dry-run
  python sagemaker_geoparquet.py
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/geoparquet"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "build_geoparquet.py",
    "validate_croissant.py",
    "entrypoint_geoparquet.sh",
]

# Croissant manifest lives outside the v24 code directory
CROISSANT_PATH = CODE_DIR.parent.parent.parent / "evidence" / "specifications" / "croissant.json"


def get_image_uri(region="us-east-1"):
    """PyTorch 2.5 CPU image (Python 3.11)."""
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.5-cpu-py311"
    )


def deploy_code(s3):
    """Upload code files + croissant.json to S3."""
    for fname in CODE_FILES:
        local = CODE_DIR / fname
        if not local.exists():
            print(f"  ERROR: {local} not found")
            sys.exit(1)
        key = f"{CODE_PREFIX}/{fname}"
        s3.upload_file(str(local), BUCKET, key)
        print(f"  Uploaded: s3://{BUCKET}/{key}")

    # Deploy croissant.json alongside code so build_geoparquet.py can find it
    if CROISSANT_PATH.exists():
        key = f"{CODE_PREFIX}/croissant.json"
        s3.upload_file(str(CROISSANT_PATH), BUCKET, key)
        print(f"  Uploaded: s3://{BUCKET}/{key}")
    else:
        print(f"  WARNING: {CROISSANT_PATH} not found -- Croissant validation will be skipped")


def main():
    parser = argparse.ArgumentParser(description="Launch geoparquet assembly SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"

    s3 = boto3.client("s3", region_name=REGION, **_aws)

    # Deploy code
    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-geoparquet-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("GeoParquet Assembly Build")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Code:     s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Data:     downloaded from S3 by build script (not mounted)")
    print(f"Output:   direct S3 upload to release/ prefix + EndOfJob backup")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] Estimated runtime: 10-20 min (TIGER download + join)")
        print(f"[DRY RUN] Estimated cost: ~$0.08 ({args.instance_type} @ $0.23/hr)")
        return

    sm = boto3.client("sagemaker", region_name=REGION, **_aws)

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
                "/opt/ml/processing/input/code/entrypoint_geoparquet.sh",
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
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": f"s3://{BUCKET}/rsct_curriculum/series_018/release/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600,
        },
    }

    print(f"\nLaunching: {job_name}")
    sm.create_processing_job(**config)
    print(f"Job created: {job_name}")
    print(f"Monitor: aws sagemaker describe-processing-job --processing-job-name {job_name} --profile {AWS_PROFILE} --region {REGION}")


if __name__ == "__main__":
    main()
