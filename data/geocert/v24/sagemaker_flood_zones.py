#!/usr/bin/env python3
"""
SageMaker Launcher: FEMA NFHL Flood Zone Enrichment

Launches a processing job to compute flood zone area fractions for all 33K ZCTAs
using FEMA's NFHL ArcGIS REST API with county-batched polygon overlay.

Instance: ml.m5.2xlarge (8 vCPU, 32 GB RAM, $0.46/hr)
Runtime:  60-120 min estimated (~3,200 county queries + spatial overlay)
Cost:     ~$0.75 estimated
Image:    PyTorch 2.5 CPU (pip installs geopandas at startup)

Mounts:
  Code:      s3://swarm-yrsn-datasets/rsct_code/geocert_v24/flood/
  Crosswalk: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet
  Tiger:     s3://swarm-yrsn-datasets/rsct_curriculum/series_018/tiger_zcta/

Output: uploaded directly to S3 by the run script (not EndOfJob-dependent).
  EndOfJob output kept as backup to /processed/ prefix.

Usage:
  python sagemaker_flood_zones.py --dry-run       # validate config
  python sagemaker_flood_zones.py                  # launch
  python sagemaker_flood_zones.py --deploy-only    # upload code+tiger, no launch
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION = "us-east-1"
AWS_PROFILE = "nsc-swarm"  # 865679935554 -- bucket + SageMaker role live here
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood"
DATA_PREFIX = "rsct_curriculum/series_018/processed"  # only crosswalk is read
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"
OUTPUT_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_zones.py",
    "entrypoint_flood.sh",
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


def deploy_tiger(s3):
    """Upload TIGER shapefile to S3 if not already there."""
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{TIGER_PREFIX}/", MaxKeys=5)
        if resp.get("KeyCount", 0) > 0:
            print(f"  TIGER already on S3: s3://{BUCKET}/{TIGER_PREFIX}/")
            return
    except Exception:
        pass

    # Look for local TIGER
    tiger_dir = Path("/tmp/tiger_zcta")
    if not tiger_dir.exists():
        tiger_dir = Path(os.environ.get("TEMP", "/tmp")) / "tiger_zcta"

    shp_files = list(tiger_dir.glob("*.*")) if tiger_dir.exists() else []
    if not shp_files:
        print(f"  WARNING: No TIGER files at {tiger_dir}")
        print("  The run script will auto-download from Census inside the container.")
        return

    for f in shp_files:
        key = f"{TIGER_PREFIX}/{f.name}"
        s3.upload_file(str(f), BUCKET, key)
        print(f"  Uploaded: {f.name} -> s3://{BUCKET}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Launch flood zone SageMaker job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deploy-only", action="store_true",
                        help="Upload code and TIGER to S3, don't launch job")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    args = parser.parse_args()

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"

    s3 = session.client("s3")

    # Deploy code
    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    print("\n=== DEPLOYING TIGER ===")
    deploy_tiger(s3)

    if args.deploy_only:
        print("\n[DEPLOY ONLY] Code and TIGER uploaded. Run without --deploy-only to launch.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-flood-{timestamp}"
    image_uri = get_image_uri(REGION)

    print("\n" + "=" * 60)
    print("FEMA NFHL Flood Zone Enrichment")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Code:     s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Xwalk:    s3://{BUCKET}/{DATA_PREFIX}")
    print(f"Tiger:    s3://{BUCKET}/{TIGER_PREFIX}/")
    print(f"Output:   direct S3 upload to s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] Estimated runtime: 60-120 min")
        print(f"[DRY RUN] Estimated cost: ~$0.75 ({args.instance_type} @ $0.46/hr)")
        return

    sm = session.client("sagemaker")

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
                "/opt/ml/processing/input/code/entrypoint_flood.sh",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 21600},  # 6 hours max
        "Environment": {
            "PYTHONUNBUFFERED": "1",
        },
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


if __name__ == "__main__":
    main()
