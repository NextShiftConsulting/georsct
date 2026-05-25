#!/usr/bin/env python3
"""
SageMaker Launcher: TWI and Watershed Features

Downloads EPA StreamCat WetIndx + Slope for all 18 CONUS HUC2 regions,
joins to the COMID-ZCTA spatial crosswalk (area-weighted), aggregates to ZCTA.

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM, $0.23/hr)
Runtime:  ~15-25 min
Cost:     ~$0.10

Inputs:
  Code:      s3://swarm-yrsn-datasets/rsct_code/geocert_v24/twi/
  Data:      s3://swarm-yrsn-datasets/rsct_curriculum/geo/comid_zcta_crosswalk.parquet
             s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet

Output:
  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/twi_features_zcta.parquet

Usage:
  python sagemaker_twi.py --dry-run
  python sagemaker_twi.py
  python sagemaker_twi.py --deploy-only
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION = "us-east-1"
AWS_PROFILE = "nsc-swarm"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/twi"
GEO_PREFIX = "rsct_curriculum/geo"
DATA_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = ["run_twi.py", "entrypoint_twi.sh"]


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
    parser = argparse.ArgumentParser(description="Launch TWI features SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deploy-only", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = session.client("s3")

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    if args.deploy_only:
        print("\n[DEPLOY ONLY] Done.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-twi-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("TWI + Watershed Features (StreamCat to ZCTA)")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Code:     s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Data:     s3://{BUCKET}/{GEO_PREFIX}/comid_zcta_crosswalk.parquet")
    print(f"Output:   s3://{BUCKET}/{DATA_PREFIX}/twi_features_zcta.parquet")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] Estimated cost: ~$0.10 ({args.instance_type} @ $0.23/hr)")
        return

    sm = session.client("sagemaker")

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 20,
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash", "/opt/ml/processing/input/code/entrypoint_twi.sh"
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
                "InputName": "comid_zcta",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{GEO_PREFIX}/comid_zcta_crosswalk.parquet",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "county_xwalk",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/zcta_county_crosswalk.parquet",
                    "LocalPath": "/opt/ml/processing/input/county",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "Environment": {"PYTHONUNBUFFERED": "1"},
    }

    response = sm.create_processing_job(**config)
    print(f"\nJob launched: {response['ProcessingJobArn']}")
    cw_group = "%2Faws%2Fsagemaker%2FProcessingJobs"
    print(f"\nMonitor:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
          f"--processing-job-name {job_name} --region {REGION} --profile nsc-swarm "
          f"--query 'ProcessingJobStatus'")
    print(f"  https://console.aws.amazon.com/cloudwatch/home?region={REGION}"
          f"#logsV2:log-groups/log-group/{cw_group}/log-events/{job_name}")
    print(f"\nStop if needed:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker stop-processing-job "
          f"--processing-job-name {job_name} --region {REGION} --profile nsc-swarm")


if __name__ == "__main__":
    main()
