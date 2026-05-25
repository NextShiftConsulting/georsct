#!/usr/bin/env python3
"""
SageMaker Launcher: FEMA NFIP Claims Enrichment

Launches a processing job to stream all ~2.4M NFIP claims from OpenFEMA,
aggregate to ZCTA level, and upload nfip_claims_zcta.parquet to S3.

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM, $0.23/hr)
Runtime:  25-40 min (standard), 30-50 min (--temporal)
Cost:     ~$0.20 estimated
Image:    PyTorch 2.5 CPU (pip installs pyarrow at startup)

Mounts:
  Code:      s3://swarm-yrsn-datasets/rsct_code/geocert_v24/nfip/
  Crosswalk: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/

No TIGER or spatial files needed — ZIP codes map directly to ZCTAs.

Output: uploaded directly to S3 by the run script (not EndOfJob-dependent).
  EndOfJob output kept as backup to /processed/ prefix.

Usage:
  python sagemaker_nfip.py --dry-run            # validate config
  python sagemaker_nfip.py                       # launch
  python sagemaker_nfip.py --deploy-only         # upload code to S3, no launch
  python sagemaker_nfip.py --max-pages 5        # quick test (50K records)
  python sagemaker_nfip.py --temporal           # also produce long + wide epoch outputs
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
AWS_PROFILE = "nsc-swarm"  # 865679935554 -- bucket + SageMaker role live here
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/nfip"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
OUTPUT_PREFIX = "rsct_curriculum/series_018/processed"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_nfip.py",
    "entrypoint_nfip.sh",
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
    parser = argparse.ArgumentParser(description="Launch NFIP claims SageMaker job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config, don't launch")
    parser.add_argument("--deploy-only", action="store_true",
                        help="Upload code to S3, don't launch job")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit pages for testing (None = all ~240 pages)")
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
    job_name = f"geocert-nfip-{timestamp}"
    image_uri = get_image_uri(REGION)

    # Build ContainerArguments
    container_args = []
    if args.max_pages:
        container_args += ["--max-pages", str(args.max_pages)]
    if args.temporal:
        container_args.append("--temporal")

    print("\n" + "=" * 60)
    print("FEMA NFIP Claims Enrichment")
    print("=" * 60)
    print(f"Job:       {job_name}")
    print(f"Instance:  {args.instance_type}")
    print(f"Image:     {image_uri}")
    print(f"Code:      s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Crosswalk: s3://{BUCKET}/{DATA_PREFIX}/")
    print(f"Output:    direct S3 upload to s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print(f"Temporal:  {'yes (long + wide epoch outputs)' if args.temporal else 'no (aggregated only)'}")
    if args.max_pages:
        print(f"TEST MODE: --max-pages {args.max_pages} (~{args.max_pages * 10_000:,} records)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        runtime = "30-50 min" if args.temporal else "25-40 min"
        print(f"[DRY RUN] Estimated runtime: {runtime}")
        print(f"[DRY RUN] Estimated cost: ~$0.20 ({args.instance_type} @ $0.23/hr)")
        if args.temporal:
            print("[DRY RUN] Temporal outputs:")
            print(f"[DRY RUN]   s3://{BUCKET}/{OUTPUT_PREFIX}/nfip_claims_long.parquet")
            print(f"[DRY RUN]   s3://{BUCKET}/{OUTPUT_PREFIX}/nfip_claims_wide.parquet")
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
                "/opt/ml/processing/input/code/entrypoint_nfip.sh",
            ],
            **({"ContainerArguments": container_args} if container_args else {}),
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},  # 2 hours max
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
