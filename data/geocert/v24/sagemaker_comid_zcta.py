#!/usr/bin/env python3
"""
SageMaker Launcher: COMID-to-ZCTA Spatial Crosswalk

Launches a processing job to build the NHDPlus COMID → ZCTA spatial crosswalk
with area fraction weights. Required for production-quality TWI aggregation.

Instance: ml.m5.12xlarge (48 vCPU, 192 GB RAM, $1.845/hr)
Runtime:  ~60-90 min (16 workers × 2179 HUC8s, ~136 HUC8s/worker)
Cost:     ~$2.80 estimated
Workers:  16 (capped; geopandas spawn workers use 4-8GB each — 48 workers OOMs)
Image:    PyTorch 2.5 CPU (pip installs geopandas + py7zr at startup)

Mounts:
  Code:  s3://swarm-yrsn-datasets/rsct_code/geocert_v24/comid_zcta/
  Tiger: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/tiger_zcta/

NHDPlus V2 catchment files downloaded directly from EDCINTL inside the
container (no pre-staging needed). VPU parquets cached to S3 after first
download so reruns skip the download.

Output: s3://swarm-yrsn-datasets/rsct_curriculum/geo/comid_zcta_crosswalk.parquet

Usage:
  python sagemaker_comid_zcta.py --dry-run
  python sagemaker_comid_zcta.py
  python sagemaker_comid_zcta.py --deploy-only
  python sagemaker_comid_zcta.py --vpus 01 02 03N   # subset for testing
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/comid_zcta"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"
OUTPUT_PREFIX = "rsct_curriculum/geo"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_comid_zcta.py",
    "entrypoint_comid_zcta.sh",
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
    parser = argparse.ArgumentParser(
        description="Launch COMID-ZCTA crosswalk SageMaker job"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deploy-only", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.12xlarge")
    parser.add_argument("--vpus", nargs="+", default=None,
                        help="Subset of VPUs (default: all 21). E.g. --vpus 12 13 15")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", region_name=REGION, **_aws)

    print("=== DEPLOYING CODE ===")
    deploy_code(s3)

    if args.deploy_only:
        print("\n[DEPLOY ONLY] Done.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geocert-comid-zcta-{timestamp}"
    image_uri = get_image_uri(REGION)

    # Pass VPU subset via environment variable
    env = {"PYTHONUNBUFFERED": "1"}
    container_cmd = ["bash", "/opt/ml/processing/input/code/entrypoint_comid_zcta.sh"]
    if args.vpus:
        env["NHDPLUS_VPUS"] = " ".join(args.vpus)

    print("\n" + "=" * 60)
    print("COMID-to-ZCTA Spatial Crosswalk")
    print("=" * 60)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    {image_uri}")
    print(f"Code:     s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Tiger:    s3://{BUCKET}/{TIGER_PREFIX}/")
    print(f"Output:   s3://{BUCKET}/{OUTPUT_PREFIX}/comid_zcta_crosswalk.parquet")
    if args.vpus:
        print(f"VPUs:     {args.vpus} (subset)")
    else:
        print(f"VPUs:     all 21 CONUS regions")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        print("[DRY RUN] Estimated runtime: ~20-30 min (21 VPUs parallel)")
        print(f"[DRY RUN] Estimated cost: ~$0.90 ({args.instance_type} @ $1.845/hr)")
        return

    sm = boto3.client("sagemaker", region_name=REGION, **_aws)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 100,   # NHDPlus archives + extracted shapefiles
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": container_cmd,
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 18000},  # 5 hours max
        "Environment": env,
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
