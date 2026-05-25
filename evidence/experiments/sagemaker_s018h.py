#!/usr/bin/env python3
"""
SageMaker Launcher: s018h Canonical YRSN Certification for CONUS-27

Mounts:
- Code:   s3://swarm-yrsn-datasets/rsct_code/series_018/s018h_canonical_certification/
- Shared: s3://swarm-yrsn-datasets/rsct_code/series_018/shared/
- Wheel:  s3://swarm-yrsn-datasets/rsct_code/wheels/20260424-052122/
- Data:   s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/
- OOF:    s3://swarm-yrsn-datasets/rsct_curriculum/series_018/oof_artifacts/
- Output: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/results/s018h/

Container layout after mount:
  /opt/ml/processing/input/code/run_s018h.py
  /opt/ml/processing/input/code/evaluate_hypotheses.py
  /opt/ml/processing/input/code/shared/       (second S3Input)
  /opt/ml/processing/input/wheels/yrsn-*.whl  (pip installed at startup)
  /opt/ml/processing/input/data/              (features parquet)
  /opt/ml/processing/input/oof/               (OOF parquets)
  /opt/ml/processing/output/                  (results JSON)

Requires yrsn wheel for canonical certificate path
(aggregate_scores_from_probs, CPGatekeeperInput, SequentialGatekeeper).
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import boto3
from swarm_auth import get_aws_credentials

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "apps"))
try:
    from instance_selector import get_image_uri
except ImportError:
    def get_image_uri(device, region="us-east-1"):
        # PyTorch 2.9.0 CPU -- latest verified tag as of 2026-04-24
        return (
            f"763104351884.dkr.ecr.{region}.amazonaws.com/"
            "pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v1.9"
        )

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.preflight import (
    preflight_check, S3Artifact, ArtifactGroup,
    wheels_group, processed_data_group, oof_artifacts_group, shared_code_group,
    BUCKET,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task subset for pilot")
    parser.add_argument("--classifier", type=str, default="mlp",
                        choices=["mlp", "logistic"])
    args = parser.parse_args()

    region = "us-east-1"
    aws_account_id = boto3.client("sts", **get_aws_credentials()).get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{aws_account_id}:role/SageMakerExecutionRole"
    bucket = "swarm-yrsn-datasets"
    code_prefix = "rsct_code/series_018/s018h_canonical_certification"
    data_prefix = "rsct_curriculum/series_018/processed"
    oof_prefix = "rsct_curriculum/series_018/oof_artifacts"
    output_prefix = "rsct_curriculum/series_018/results/s018h"
    wheel_prefix = "rsct_code/wheels/20260424-052122"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"s018h-canonical-cert-{timestamp}"

    image_uri = get_image_uri("cpu", region)

    # Build container arguments
    container_args = [
        "--oof-dir", "/opt/ml/processing/input/oof",
        "--data-dir", "/opt/ml/processing/input/data",
        "--output-dir", "/opt/ml/processing/output",
        "--classifier", args.classifier,
    ]
    if args.tasks:
        container_args.extend(["--tasks", args.tasks])

    ok = preflight_check([
        wheels_group(),
        processed_data_group(),
        oof_artifacts_group(),
        shared_code_group(),
        ArtifactGroup("s018h code", [
            S3Artifact(BUCKET, "rsct_code/series_018/s018h_canonical_certification/run_s018h.py"),
            S3Artifact(BUCKET, "rsct_code/series_018/s018h_canonical_certification/evaluate_hypotheses.py"),
        ]),
    ], region=region, dry_run=args.dry_run)
    if not ok:
        print("\nFix missing artifacts before launching. Job NOT started.")
        sys.exit(1)

    print("=" * 60)
    print("s018h: Canonical YRSN Certification for CONUS-27")
    print("=" * 60)
    print(f"Job:        {job_name}")
    print(f"Instance:   {args.instance_type}")
    print(f"Code:       s3://{bucket}/{code_prefix}/")
    print(f"Data:       s3://{bucket}/{data_prefix}/")
    print(f"OOF:        s3://{bucket}/{oof_prefix}/")
    print(f"Output:     s3://{bucket}/{output_prefix}/")
    print(f"Wheel:      s3://{bucket}/{wheel_prefix}/")
    print(f"Classifier: {args.classifier}")
    if args.tasks:
        print(f"Tasks:      {args.tasks}")
    else:
        print(f"Tasks:      ALL 27")
    print("=" * 60)

    if args.dry_run:
        n_tasks = len(args.tasks.split(",")) if args.tasks else 27
        n_classifiers = n_tasks * 3 * 2 * 2  # tasks x families x (real+random) x (train+eval)
        print(f"\n[DRY RUN] Would launch with above config.")
        print(f"[DRY RUN] ~{n_classifiers} classifiers, estimated runtime: ~{n_classifiers * 2}s")
        return

    sm = boto3.client("sagemaker", region_name=region)

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
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "/bin/bash", "-c",
                "pip install /opt/ml/processing/input/wheels/yrsn-*.whl && "
                "export PYTHONPATH=/opt/ml/processing/input/code:$PYTHONPATH && "
                "python3 /opt/ml/processing/input/code/run_s018h.py "
                + " ".join(container_args),
            ],
        },
        "RoleArn": role_arn,
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{code_prefix}/",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "shared_code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/rsct_code/series_018/shared/",
                    "LocalPath": "/opt/ml/processing/input/code/shared",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "wheels",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{wheel_prefix}/",
                    "LocalPath": "/opt/ml/processing/input/wheels",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{data_prefix}/",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "oof",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{oof_prefix}/",
                    "LocalPath": "/opt/ml/processing/input/oof",
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
                        "S3Uri": f"s3://{bucket}/{output_prefix}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
        "Environment": {"PYTHONUNBUFFERED": "1"},
    }

    response = sm.create_processing_job(**config)

    print(f"\nJob launched: {response['ProcessingJobArn']}")
    print(f"\nMonitor:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
          f"--processing-job-name {job_name}")
    print(f"  MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs "
          f"--log-stream-name {job_name}/algo-1-stdout --follow")


if __name__ == "__main__":
    main()
