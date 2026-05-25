#!/usr/bin/env python3
"""
SageMaker Launcher: NOAA Storm Events Flood Enrichment

Launches a processing job to download NOAA NCEI Storm Events (1996-2024),
extract flood events, aggregate to ZCTA, and upload to S3.

Instance: ml.m5.large (2 vCPU, 8 GB RAM, $0.115/hr)
Runtime:  15-25 min estimated (29 annual gzip files)
Cost:     ~$0.06 estimated
Image:    PyTorch 2.5 CPU (pip installs pyarrow at startup)

Mounts:
  Code:      s3://swarm-yrsn-datasets/rsct_code/geocert_v24/noaa/
  Crosswalk: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/

Output: uploaded directly to S3 by the run script (not EndOfJob-dependent).
  EndOfJob output kept as backup to /processed/ prefix.

Usage:
  python sagemaker_noaa.py --dry-run       # validate config
  python sagemaker_noaa.py                  # launch
  python sagemaker_noaa.py --deploy-only    # upload code to S3, no launch
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/noaa"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
OUTPUT_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_noaa.py",
    "entrypoint_noaa.sh",
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
    parser = argparse.ArgumentParser(description="Launch NOAA storm events SageMaker job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config, don't launch")
    parser.add_argument("--deploy-only", action="store_true",
                        help="Upload code to S3, don't launch job")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--temporal", action="store_true",
                        help="Also produce long (zcta x year) and wide (epoch) Parquet outputs")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"

    s3 = boto3.client("s3", **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    if args.deploy_only:
        print("\n[DEPLOY ONLY] Code uploaded. Run without --deploy-only to launch.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-noaa-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("NOAA Storm Events Flood Enrichment")
    print("=" * 60)
    print(f"Job:       {job_name}")
    print(f"Instance:  {args.instance_type}")
    print(f"Image:     {image_uri}")
    print(f"Code:      s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Crosswalk: s3://{BUCKET}/{DATA_PREFIX}/")
    print(f"Output:    direct S3 upload to s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print(f"Temporal:  {'yes (long + wide epoch outputs)' if args.temporal else 'no (aggregated only)'}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        runtime = "20-35 min" if args.temporal else "15-25 min"
        print(f"[DRY RUN] Estimated runtime: {runtime}")
        print(f"[DRY RUN] Estimated cost: ~$0.06 ({args.instance_type} @ $0.115/hr)")
        if args.temporal:
            print("[DRY RUN] Temporal outputs:")
            print(f"[DRY RUN]   s3://{BUCKET}/{OUTPUT_PREFIX}/noaa_storm_events_long.parquet")
            print(f"[DRY RUN]   s3://{BUCKET}/{OUTPUT_PREFIX}/noaa_storm_events_wide.parquet")
        return

    sm = boto3.client("sagemaker", **_aws)

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
                "bash",
                "/opt/ml/processing/input/code/entrypoint_noaa.sh",
            ],
            "ContainerArguments": ["--temporal"] if args.temporal else [],
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
                "InputName": "data",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/zcta_county_crosswalk.parquet",
                    "LocalPath": "/opt/ml/processing/input/data",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},  # 1 hour max
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
