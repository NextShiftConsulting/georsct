#!/usr/bin/env python3
"""
SageMaker Launcher: Combined Stage 2+3 — Extract + Overlay in one pass.

Reads raw FEMA zips from S3 flood_raw/, extracts S_FLD_HAZ_AR shapefiles,
overlays against TIGER ZCTA boundaries, outputs flood_zones_zcta.parquet.

Instance: ml.m5.12xlarge (48 vCPU, 192 GB, 10 Gbps)
Image:    sklearn 1.2-1 (no PyTorch needed — saves ~8 GB pull time)
Volume:   100 GB (raw zips ~86 GB streamed through temp, not all at once)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_combined"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_combined.py",
    "entrypoint_flood_combined.sh",
]


def get_image_uri(region="us-east-1"):
    # PyTorch CPU image: Python 3.11, modern numpy/pandas — geopandas installs cleanly
    # sklearn image has Python 3.9 + pinned numpy 1.24 + pandas 1.1.3 which segfaults
    # on numpy upgrade needed by geopandas
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


def count_raw_zips(s3):
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{DATA_PREFIX}/flood_raw/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".zip"):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Launch combined flood extract+overlay")
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
    job_name = f"geocert-flood-combined-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Combined Extract+Overlay")
    print("=" * 60)
    print(f"Job:       {job_name}")
    print(f"Instance:  {args.instance_type}")
    print(f"Image:     {image_uri}")
    print(f"Input:     s3://{BUCKET}/{DATA_PREFIX}/flood_raw/ ({n_raw} zips)")
    print(f"TIGER:     s3://{BUCKET}/{TIGER_PREFIX}/")
    print(f"Output:    s3://{BUCKET}/{DATA_PREFIX}/flood_zones_zcta.parquet")
    print(f"Validation: 7 checks (CRS, bounds, geometry, winding, Harris, area conservation, distribution)")
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
                "/opt/ml/processing/input/code/entrypoint_flood_combined.sh",
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
