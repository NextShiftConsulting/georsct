#!/usr/bin/env python3
"""
SageMaker Launcher: Stage 2 — Extract S_FLD_HAZ_AR from raw FEMA zips.

Reads zips from flood_raw/, extracts shapefiles, writes JSON to flood_fetch/.
Includes validation (geometry, winding, zip integrity, feature counts).

Instance: ml.m5.12xlarge (48 vCPU, 192 GB, 10 Gbps — max parallelism)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_extract"
DATA_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_extract.py",
    "entrypoint_flood_extract.sh",
]


def get_image_uri(region="us-east-1"):
    return (
        f"683313688378.dkr.ecr.{region}.amazonaws.com/"
        "sagemaker-scikit-learn:1.2-1-cpu-py3"
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


def count_raw_zips(s3):
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{DATA_PREFIX}/flood_raw/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".zip"):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Launch flood extract (Stage 2)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.12xlarge")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", region_name=REGION, **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    print("\n=== PRE-FLIGHT CHECK ===")
    n_raw = count_raw_zips(s3)
    print(f"  Raw zips on S3: {n_raw}")
    if n_raw == 0:
        print("  ERROR: No raw zips found. Run Stage 1 (bulk download) first.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-flood-extract-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Extract (Stage 2)")
    print("=" * 60)
    print(f"Job:       {job_name}")
    print(f"Instance:  {args.instance_type}")
    print(f"Image:     {image_uri}")
    print(f"Input:     s3://{BUCKET}/{DATA_PREFIX}/flood_raw/ ({n_raw} zips)")
    print(f"Output:    s3://{BUCKET}/{DATA_PREFIX}/flood_fetch/")
    print(f"Validation: zip integrity, geometry, winding, feature counts")
    print("=" * 60)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch with above config.")
        return

    sm = boto3.client("sagemaker", region_name=REGION, **_aws)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 100,
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash",
                "/opt/ml/processing/input/code/entrypoint_flood_extract.sh",
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
                        "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 21600},
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
