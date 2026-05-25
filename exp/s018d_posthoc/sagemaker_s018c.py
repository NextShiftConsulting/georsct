#!/usr/bin/env python3
"""
SageMaker Launcher: S018C Substrate Expansion (ACS-only vs full geospatial).

Mounts:
  - zcta_features_labels_with_lags.parquet  (processed_with_lags/)
  - oof_*.parquet                           (oof_artifacts/)
  - shared/ Python modules                  (rsct_code/series_018/shared/)
  - yrsn + yrsn_controlplane wheels         (rsct_code/wheels/)
  - run_s018c_normalized.py                 (uploaded inline)

The entrypoint installs wheels, sets PYTHONPATH, and runs the script.
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.preflight import (
    preflight_check,
    wheels_group,
    oof_artifacts_group,
    shared_code_group,
    S3Artifact,
    ArtifactGroup,
    BUCKET,
    WHEEL_PREFIX,
    SHARED_CODE_PREFIX,
    OOF_PREFIX,
)

REGION = "us-east-1"
AWS_PROFILE = "nsc-swarm"
INSTANCE_TYPE = "ml.m5.2xlarge"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker"

DATA_PREFIX = "rsct_curriculum/series_018/processed_with_lags"
CODE_PREFIX = "rsct_code/series_018/s018c_normalized"
OUTPUT_PREFIX = "rsct_curriculum/series_018/results/s018c_normalized"


def data_with_lags_group() -> ArtifactGroup:
    return ArtifactGroup("processed data (with lags)", [
        S3Artifact(BUCKET, f"{DATA_PREFIX}/zcta_features_labels_with_lags.parquet"),
    ])


def upload_code(session, dry_run: bool = False):
    """Upload run_s018c_normalized.py and bootstrap.sh to S3."""
    s3 = session.client("s3")
    parent = Path(__file__).parent
    for fname in ("run_s018c_normalized.py", "bootstrap.sh"):
        local = parent / fname
        s3_key = f"{CODE_PREFIX}/{fname}"
        if dry_run:
            print(f"  [DRY RUN] Would upload {fname} -> s3://{BUCKET}/{s3_key}")
        else:
            s3.upload_file(str(local), BUCKET, s3_key)
            print(f"  Uploaded {fname} -> s3://{BUCKET}/{s3_key}")


def main():
    parser = argparse.ArgumentParser(description="Launch s018c normalized substrate expansion")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    args = parser.parse_args()

    session = boto3.Session(profile_name=AWS_PROFILE)
    account_id = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"s018c-norm-{timestamp}"

    print("=" * 70)
    print("S018C: Substrate Expansion -- ACS-only vs Full Geospatial")
    print("=" * 70)
    print(f"Job:      {job_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Image:    pytorch 2.8.0 CPU py312")
    print(f"Data:     s3://{BUCKET}/{DATA_PREFIX}/")
    print(f"OOF:      s3://{BUCKET}/{OOF_PREFIX}/")
    print(f"Shared:   s3://{BUCKET}/{SHARED_CODE_PREFIX}/")
    print(f"Wheels:   s3://{BUCKET}/{WHEEL_PREFIX}/")
    print(f"Output:   s3://{BUCKET}/{OUTPUT_PREFIX}/{job_name}/")
    print("=" * 70)

    # Preflight
    ok = preflight_check(
        [
            data_with_lags_group(),
            oof_artifacts_group(),
            shared_code_group(),
            wheels_group(require_controlplane=True),
        ],
        region=REGION,
        dry_run=args.dry_run,
    )
    if not ok:
        print("\nFix missing artifacts before launching.")
        sys.exit(1)

    # Upload code files
    print("\nUploading code...")
    upload_code(session, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with above config.")
        return

    sm = session.client("sagemaker", region_name=REGION)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/code/bootstrap.sh",
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
                "InputName": "data",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "oof",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{OOF_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/oof",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "shared_code",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{SHARED_CODE_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/shared_code/shared",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "wheels",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{WHEEL_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/wheels",
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
                        "S3Uri": f"s3://{BUCKET}/{OUTPUT_PREFIX}/{job_name}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        },
    }

    response = sm.create_processing_job(**config)
    print(f"\nJob launched: {response['ProcessingJobArn']}")
    print(f"\n=== EXPERIMENT LAUNCHED ===")
    print(f"Job:     {job_name}")
    print(f"Status:  InProgress")
    print(f"ETA:     ~15 min")
    print(f"\nMonitor:")
    print(f"  AWS_PROFILE=nsc-swarm aws sagemaker describe-processing-job --processing-job-name {job_name} --query '{{Status:ProcessingJobStatus}}'")
    print(f"\nLogs:")
    print(f"  MSYS_NO_PATHCONV=1 AWS_PROFILE=nsc-swarm aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix {job_name} --follow")
    print(f"\nOutput: s3://{BUCKET}/{OUTPUT_PREFIX}/{job_name}/")
    print(f"\nCloudWatch:")
    cw_url = f"https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FProcessingJobs/log-events/{job_name}"
    print(f"  {cw_url}")
    print(f"\n=== YOU CAN CLOSE YOUR LAPTOP ===")


if __name__ == "__main__":
    main()
