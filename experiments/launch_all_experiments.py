#!/usr/bin/env python3
"""Unified SageMaker launcher for all 3 GeoRSCT paper experiments.

Fires S018B-FREEZE, F001-KAPPA-VS-SIMPLEX, and S018C2-RESIDUAL-NORM-CONTROL
as parallel SageMaker Processing Jobs on ml.m5.xlarge.

Usage:
    python launch_all_experiments.py --dry-run   # validate only
    python launch_all_experiments.py             # launch all 3

Requirements:
    - AWS credentials via IAM role or ~/.aws/credentials (profile: nsc-swarm)
    - boto3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3

# -- config --
REGION = "us-east-1"
ROLE_ARN = None  # auto-discover from SageMaker default
INSTANCE_TYPE = "ml.m5.xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 30
MAX_RUNTIME_SECONDS = 3600  # 1 hour safety cap

# SageMaker sklearn image (us-east-1)
SKLEARN_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# Experiment definitions
EXPERIMENTS = [
    {
        "name": "s018b-freeze",
        "job_prefix": "geocert-s018b-freeze",
        "script_path": os.path.join(os.path.dirname(__file__),
                                     "s018b_freeze", "job_files", "run_s018b_freeze.py"),
        "description": "Benchmark dataset freeze and reproducibility audit",
    },
    {
        "name": "f001-kappa-vs-simplex",
        "job_prefix": "geocert-f001-kappa-simplex",
        "script_path": os.path.join(os.path.dirname(__file__),
                                     "f001_kappa_vs_simplex", "job_files", "run_f001.py"),
        "description": "Scalar kappa vs R/S/N simplex evaluation",
    },
    {
        "name": "s018c2-residual-norm",
        "job_prefix": "geocert-s018c2-resid-norm",
        # S018C2 lives in yrsn-experiments repo
        "script_path": os.path.normpath(os.path.join(
            os.path.expanduser("~"), "github",
            "yrsn-experiments", "exp", "series_018",
            "s018c2_residual_norm_control", "job_files", "run_s018c2.py")),
        "description": "Certificate proxy residual normalization",
    },
]


def get_execution_role(session):
    """Get SageMaker execution role from the account."""
    iam = session.client("iam")
    # Try common SageMaker role names
    for role_name in ["SageMakerExecutionRole", "AmazonSageMaker-ExecutionRole",
                      "SageMakerRole", "sagemaker-execution-role"]:
        try:
            resp = iam.get_role(RoleName=role_name)
            return resp["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            continue

    # List roles with SageMaker in trust policy
    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page["Roles"]:
            trust = json.dumps(role.get("AssumeRolePolicyDocument", {}))
            if "sagemaker" in trust.lower():
                return role["Arn"]

    raise RuntimeError("No SageMaker execution role found. Set ROLE_ARN manually.")


def validate_script(exp):
    """Check that the run script exists."""
    path = exp["script_path"]
    if not os.path.isfile(path):
        # Try alternative path resolution
        alt = os.path.abspath(path)
        if not os.path.isfile(alt):
            return False, f"Script not found: {path}"
    return True, "OK"


def launch_processing_job(sm_client, exp, role_arn, dry_run=False):
    """Launch a single SageMaker Processing Job."""
    job_name = f"{exp['job_prefix']}-{TIMESTAMP}"
    script_path = os.path.abspath(exp["script_path"])

    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']}")
    print(f"Job name:   {job_name}")
    print(f"Script:     {script_path}")
    print(f"Instance:   {INSTANCE_TYPE}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would launch {job_name}")
        return job_name, "DRY_RUN"

    # Read script content for inline code
    with open(script_path, "r") as f:
        script_content = f.read()

    # Upload script to S3
    s3 = boto3.client("s3", region_name=REGION)
    script_key = f"geocert-experiments/scripts/{job_name}/run.py"
    s3.put_object(
        Bucket="swarm-yrsn-datasets",
        Key=script_key,
        Body=script_content.encode("utf-8"),
    )
    script_s3_uri = f"s3://swarm-yrsn-datasets/{script_key}"
    print(f"  Script uploaded to {script_s3_uri}")

    try:
        sm_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": INSTANCE_COUNT,
                    "InstanceType": INSTANCE_TYPE,
                    "VolumeSizeInGB": VOLUME_SIZE_GB,
                }
            },
            AppSpecification={
                "ImageUri": SKLEARN_IMAGE,
                "ContainerEntrypoint": ["python3", "-u", "/opt/ml/processing/input/run.py"],
            },
            ProcessingInputs=[
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": script_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                }
            ],
            RoleArn=role_arn,
            Environment={
                "INSTANCE_TYPE": INSTANCE_TYPE,
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS,
            },
            Tags=[
                {"Key": "project", "Value": "geocert"},
                {"Key": "experiment", "Value": exp["name"]},
                {"Key": "paper", "Value": "neurips-2026"},
            ],
        )
        print(f"  LAUNCHED: {job_name}")
        return job_name, "InProgress"
    except Exception as e:
        print(f"  FAILED: {e}")
        return job_name, f"FAILED: {e}"


def check_existing_jobs(sm_client):
    """Check for already-running geocert jobs to avoid duplicates."""
    resp = sm_client.list_processing_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
        StatusEquals="InProgress",
    )
    running = []
    for job in resp.get("ProcessingJobSummaries", []):
        if job["ProcessingJobName"].startswith("geocert-"):
            running.append(job["ProcessingJobName"])
    return running


def main():
    parser = argparse.ArgumentParser(description="Launch GeoRSCT experiments on SageMaker")
    parser.add_argument("--dry-run", action="store_true", help="Validate without launching")
    parser.add_argument("--profile", default="nsc-swarm", help="AWS profile (default: nsc-swarm)")
    args = parser.parse_args()

    print("=" * 60)
    print("GeoRSCT Unified Experiment Launcher")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Profile:   {args.profile}")
    print(f"Dry run:   {args.dry_run}")
    print("=" * 60)

    # Validate all scripts exist
    all_valid = True
    for exp in EXPERIMENTS:
        ok, msg = validate_script(exp)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {exp['name']}: {exp['script_path']}")
        if not ok:
            print(f"       {msg}")
            all_valid = False

    if not all_valid:
        print("\nFATAL: Some scripts missing. Fix paths before launching.")
        sys.exit(1)

    # AWS session
    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    sm_client = session.client("sagemaker")

    # Check for duplicate jobs
    running = check_existing_jobs(sm_client)
    if running:
        print(f"\nWARNING: {len(running)} geocert jobs already running:")
        for j in running:
            print(f"  - {j}")
        if not args.dry_run:
            print("Proceeding anyway (different timestamps)...")

    # Get execution role
    role_arn = ROLE_ARN
    if not role_arn:
        try:
            role_arn = get_execution_role(session)
            print(f"\nExecution role: {role_arn}")
        except Exception as e:
            print(f"\nFATAL: Cannot find SageMaker role: {e}")
            sys.exit(1)

    # Launch all experiments
    results = []
    for exp in EXPERIMENTS:
        job_name, status = launch_processing_job(sm_client, exp, role_arn, dry_run=args.dry_run)
        results.append({"experiment": exp["name"], "job_name": job_name, "status": status})

    # Summary
    print("\n" + "=" * 60)
    print("LAUNCH SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['experiment']:30s} {r['status']:12s} {r['job_name']}")

    if not args.dry_run:
        print(f"\nMonitor with:")
        print(f"  aws sagemaker list-processing-jobs --profile {args.profile} "
              f"--sort-by CreationTime --sort-order Descending --max-results 5 "
              f"--query \"ProcessingJobSummaries[*].[ProcessingJobName,ProcessingJobStatus]\" "
              f"--output table")

    print("\nDone.")


if __name__ == "__main__":
    main()
