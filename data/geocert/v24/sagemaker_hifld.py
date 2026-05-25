#!/usr/bin/env python3
"""
SageMaker Launcher: HIFLD Hospital & Pharmacy Feature Build

Launches a processing job to compute per-ZCTA hospital and pharmacy access
features using HIFLD 2020 Hospitals CSV (beds, trauma) and CDC Vaccines.gov
pharmacy locations.

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM, $0.23/hr)
Runtime:  15-30 min estimated (haversine distances: 31K ZCTAs x 7K hospitals + 41K pharmacies)
Cost:     ~$0.12 estimated

Mounts:
  Code:   s3://swarm-yrsn-datasets/rsct_code/geocert_v24/hifld/
  HIFLD:  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/source/
  Data:   s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/

Output: uploaded directly to S3 by the run script.

Usage:
  python sagemaker_hifld.py --dry-run
  python sagemaker_hifld.py
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/hifld"
SOURCE_PREFIX = "rsct_curriculum/series_018/source"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
OUTPUT_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "build_hifld_features.py",
    "entrypoint_hifld.sh",
]


def get_image_uri(region="us-east-1"):
    """PyTorch 2.5 CPU image (Python 3.11)."""
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.5-cpu-py311"
    )


def deploy_code(s3):
    """Upload code files to S3."""
    for fname in CODE_FILES:
        local = CODE_DIR / fname
        if not local.exists():
            print(f"  ERROR: {local} not found")
            sys.exit(1)
        key = f"{CODE_PREFIX}/{fname}"
        s3.upload_file(str(local), BUCKET, key)
        print(f"  Uploaded: s3://{BUCKET}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Launch HIFLD SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"

    s3 = boto3.client("s3", **_aws)

    # Deploy code
    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    # Verify HIFLD CSV is on S3
    hifld_key = f"{SOURCE_PREFIX}/HIFLD_2022_Hospitals.csv"
    try:
        s3.head_object(Bucket=BUCKET, Key=hifld_key)
        print(f"  HIFLD CSV verified: s3://{BUCKET}/{hifld_key}")
    except Exception:
        print(f"  ERROR: HIFLD CSV not found at s3://{BUCKET}/{hifld_key}")
        print("  Upload with: aws s3 cp V3/data/HIFLD_2022_Hospitals.csv s3://...")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-hifld-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("HIFLD Hospital & Pharmacy Feature Build")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Code:     s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"HIFLD:    s3://{BUCKET}/{SOURCE_PREFIX}/")
    print(f"Data:     s3://{BUCKET}/{DATA_PREFIX}/")
    print(f"Output:   direct S3 upload + EndOfJob backup")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] Estimated runtime: 15-30 min")
        print(f"[DRY RUN] Estimated cost: ~$0.12 ({args.instance_type} @ $0.23/hr)")
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
                "/opt/ml/processing/input/code/entrypoint_hifld.sh",
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
                "InputName": "hifld",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{SOURCE_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/hifld",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/zcta_features_labels.parquet",
                    "LocalPath": "/opt/ml/processing/input/data/zcta_features_labels.parquet",
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
