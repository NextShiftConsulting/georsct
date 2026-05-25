#!/usr/bin/env python3
"""
SageMaker Launcher: S019A Certificate Invariance Gradient

7 embeddings x 3 solvers x 3 targets = 63 cells x 5 folds = 315 fits.
Theory kappa D*/D + paired flat/Oobleck gate evaluation.

Mounts:
- Code:    s3://swarm-yrsn-datasets/rsct_code/series_019/s019a/
- Shared:  s3://swarm-yrsn-datasets/rsct_code/series_019/shared/
- Wheel:   s3://swarm-yrsn-datasets/rsct_code/wheels/{latest}/
- Data:    s3://swarm-yrsn-datasets/geocert/v23.0.2/georsct_table.parquet
- Repr:    s3://swarm-yrsn-datasets/rsct_curriculum/series_018/artifacts/representations/
- Output:  s3://swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019a/
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "series_018"))

from shared.preflight import (
    preflight_check,
    wheels_group,
    S3Artifact,
    ArtifactGroup,
    BUCKET,
    WHEEL_PREFIX,
)

# ---------------------------------------------------------------------------
# S3 paths
# ---------------------------------------------------------------------------

CODE_PREFIX = "rsct_code/series_019/s019a"
SHARED_PREFIX = "rsct_code/series_019/shared"
GEOCERT_PREFIX = "geocert/v23.0.2"
OUTPUT_PREFIX_BASE = "rsct_curriculum/series_019/results/s019a"

# Two representation trees — canonical (mirrors HF) and legacy (experiment artifacts)
REPR_TREES = {
    "v23": f"{GEOCERT_PREFIX}/representations",
    "legacy": "rsct_curriculum/series_018/artifacts/representations",
}


def get_image_uri(region="us-east-1"):
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v1.9"
    )


def s019a_code_group() -> ArtifactGroup:
    return ArtifactGroup("s019a code", [
        S3Artifact(BUCKET, f"{CODE_PREFIX}/run_s019a.py"),
        S3Artifact(BUCKET, f"{CODE_PREFIX}/bootstrap.sh"),
        # s019a imports certify_group from shared.theory_certifier
        S3Artifact(BUCKET, f"{SHARED_PREFIX}/__init__.py"),
        S3Artifact(BUCKET, f"{SHARED_PREFIX}/theory_certifier.py"),
        S3Artifact(BUCKET, f"{SHARED_PREFIX}/constants.py"),
    ])


def s019a_data_group() -> ArtifactGroup:
    return ArtifactGroup("s019a data", [
        S3Artifact(BUCKET, f"{GEOCERT_PREFIX}/georsct_table.parquet",
                   required_columns=["target_diabetes", "target_population_density",
                                     "target_elevation", "county_name"]),
    ])


def s019a_repr_group(repr_prefix: str) -> ArtifactGroup:
    return ArtifactGroup("s019a representations", [
        S3Artifact(BUCKET, f"{repr_prefix}/pca32_v1.npz",
                   required_keys=["scaler_mean", "scaler_scale",
                                  "pca_components", "pca_mean", "feature_schema"]),
        S3Artifact(BUCKET, f"{repr_prefix}/spatial_lag_v1.npz",
                   required_keys=["scaler_mean", "scaler_scale",
                                  "pca_components", "pca_mean", "feature_schema"]),
        S3Artifact(BUCKET, f"{repr_prefix}/zcta_latents_v1.npz",
                   required_keys=["latents"]),
        S3Artifact(BUCKET, f"{repr_prefix}/gnn_v2_latents.npz",
                   required_keys=["Z", "zcta_id"]),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Launch S019A Certificate Invariance Gradient on SageMaker"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Pre-flight check only -- do not launch job")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge",
                        help="SageMaker instance type (default: ml.m5.2xlarge)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip S3 artifact pre-flight")
    parser.add_argument("--repr-source", choices=list(REPR_TREES.keys()),
                        default="v23",
                        help="Representation tree: 'v23' (canonical, mirrors HF) "
                             "or 'legacy' (experiment artifacts). Default: v23")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42). "
                             "Output written to s019a/seed_{SEED}/")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Launch three parallel jobs with seeds 42, 123, 456")
    args = parser.parse_args()

    CROSS_SEEDS = [42, 123, 456]
    seeds = CROSS_SEEDS if args.all_seeds else [args.seed]
    repr_prefix = REPR_TREES[args.repr_source]

    region = "us-east-1"
    session = boto3.Session(profile_name="nsc-swarm", region_name=region)
    aws_account_id = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{aws_account_id}:role/SageMakerExecutionRole"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    image_uri = get_image_uri(region)

    # Pre-flight (once, shared across seeds)
    if not args.skip_preflight:
        artifacts = [
            wheels_group(),
            s019a_code_group(),
            s019a_data_group(),
            s019a_repr_group(repr_prefix),
        ]
        ok = preflight_check(artifacts, region=region, dry_run=args.dry_run)
        if not ok:
            print("\nFix missing artifacts before launching. Job NOT started.")
            sys.exit(1)
    else:
        print("[WARNING] Pre-flight check skipped.")

    if args.dry_run:
        for seed in seeds:
            output_prefix = f"{OUTPUT_PREFIX_BASE}/seed_{seed}"
            print(f"\n[DRY RUN] Seed {seed}: would launch s019a-cert-invariance-s{seed}-{timestamp}")
            print(f"[DRY RUN]   Output -> s3://{BUCKET}/{output_prefix}/")
            print(f"[DRY RUN]   Env: S019A_SEED={seed}")
        print(f"\n[DRY RUN] {len(seeds)} job(s), 315 fits each, estimated ~15-20 min on {args.instance_type}")
        return

    sm = session.client("sagemaker")
    launched = []
    for seed in seeds:
        job_name = launch_one_job(
            sm, seed, timestamp, args.instance_type,
            repr_prefix, args.repr_source, image_uri,
            role_arn, region,
        )
        launched.append((seed, job_name))

    if len(launched) > 1:
        print("\n" + "=" * 60)
        print(f"Launched {len(launched)} jobs:")
        for seed, name in launched:
            print(f"  seed {seed}: {name}")
        print("=" * 60)


def launch_one_job(
    sm, seed: int, timestamp: str, instance_type: str,
    repr_prefix: str, repr_source: str, image_uri: str,
    role_arn: str, region: str,
):
    """Launch a single S019A processing job for one seed."""
    job_name = f"s019a-cert-invariance-s{seed}-{timestamp}"
    output_prefix = f"{OUTPUT_PREFIX_BASE}/seed_{seed}"

    container_args = [
        "--data-dir", "/opt/ml/processing/input/data",
        "--repr-dir", "/opt/ml/processing/input/repr",
        "--output-dir", "/opt/ml/processing/output",
    ]

    print("=" * 60)
    print("S019A: Certificate Invariance Gradient")
    print("=" * 60)
    print(f"Job:        {job_name}")
    print(f"Seed:       {seed}")
    print(f"Instance:   {instance_type}")
    print(f"Targets:    diabetes, population_density, elevation")
    print(f"Embeddings: PCA32, Spatial Lag, GNN, ACS raw, Geo-spatial, Noisy control, Domain features")
    print(f"Solvers:    HistGBDT, Ridge, MLP")
    print(f"Cells:      63 (7x3x3)")
    print(f"Fits:       315 (63 cells x 5 folds)")
    print(f"Kappa:      theory D*/D (RegressionKappaEvaluator, LOO)")
    print(f"Gates:      paired flat (CONUS27) + Oobleck (ADR-023 sigmoidal)")
    print(f"Sigma:      compute_sigma_request(N)")
    print(f"Repr:       s3://{BUCKET}/{repr_prefix}/ ({repr_source})")
    print(f"Code:       s3://{BUCKET}/{CODE_PREFIX}/")
    print(f"Shared:     s3://{BUCKET}/{SHARED_PREFIX}/")
    print(f"Output:     s3://{BUCKET}/{output_prefix}/")
    print("=" * 60)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/code/bootstrap.sh",
            ],
            "ContainerArguments": container_args,
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
                "InputName": "wheels",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{WHEEL_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/wheels",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "features",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{GEOCERT_PREFIX}/georsct_table.parquet",
                    "LocalPath": "/opt/ml/processing/input/data/georsct_table.parquet",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "shared",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{SHARED_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/code/shared",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "repr",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{repr_prefix}/",
                    "LocalPath": "/opt/ml/processing/input/repr",
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
                        "S3Uri": f"s3://{BUCKET}/{output_prefix}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 21600},  # 6 hr safety
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "S019A_SEED": str(seed),
        },
    }

    response = sm.create_processing_job(**config)

    print(f"\nJob launched: {response['ProcessingJobArn']}")
    print(f"\nMonitor:")
    print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
          f"--processing-job-name {job_name} --region {region} --profile nsc-swarm")
    print(f"  MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs "
          f"--log-stream-name-prefix {job_name} --follow --region {region} --profile nsc-swarm")
    print(f"\nResults will be at:")
    print(f"  s3://{BUCKET}/{output_prefix}/")

    return job_name


if __name__ == "__main__":
    main()
