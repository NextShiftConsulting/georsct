#!/usr/bin/env python3
"""
SageMaker Launcher: FEMA NFHL Bulk GDB Download

Downloads county-level file geodatabases directly from FEMA, extracts
S_Fld_Haz_Ar features, saves as JSON to S3. ~2,654 CONUS counties.

Much faster than ArcGIS REST API: direct HTTP download vs paginated queries.
Estimated: 1.5-3 hours for all of CONUS.

Instance: ml.m5.2xlarge (32 GB — GDB extraction needs headroom)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_bulk"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_bulk_download.py",
    "entrypoint_flood_bulk.sh",
    "nfhl_download_catalog.json",
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


def main():
    parser = argparse.ArgumentParser(description="Launch bulk NFHL download job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-flood-bulk-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Bulk GDB Download")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Method:   Direct county GDB download from FEMA")
    print(f"Output:   s3://{BUCKET}/{DATA_PREFIX}/flood_raw/")
    print("=" * 60)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] Estimated: 1.5-3 hours, ~$1.50")
        return

    sm = boto3.client("sagemaker", **_aws)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 30,  # just temp zip storage
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash",
                "/opt/ml/processing/input/code/entrypoint_flood_bulk.sh",
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
