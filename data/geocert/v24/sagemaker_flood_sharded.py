#!/usr/bin/env python3
"""
SageMaker Launcher: Sharded Flood Pipeline

Orchestrates parallel SageMaker processing jobs for the county-level
flood overlay pipeline. Scans S3 for completed counties, deduplicates
raw FEMA zips by FIPS (latest date wins), splits remaining work into
N shards, and launches one job per shard.

Runs locally on Windows (Git Bash). Each shard gets its own county
list JSON uploaded to S3.

Instance: ml.m5.4xlarge (16 vCPU, 64 GB per shard)
Runtime:  up to 6 hours per shard
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from swarm_auth import get_aws_credentials

REGION = "us-east-1"
BUCKET = "swarm-yrsn-datasets"
CODE_PREFIX = "rsct_code/geocert_v24/flood_shard"
DATA_PREFIX = "rsct_curriculum/series_018/processed"
TIGER_PREFIX = "rsct_curriculum/series_018/tiger_zcta"

CODE_DIR = Path(__file__).parent
CODE_FILES = [
    "run_flood_shard.py",
    "entrypoint_flood_shard.sh",
]

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}


def get_image_uri(region: str = "us-east-1") -> str:
    """Return the PyTorch CPU container image URI."""
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.5-cpu-py311"
    )


def _s3_list_prefix(
    s3: Any,
    prefix: str,
    suffix: str = "",
) -> list[tuple[str, str]]:
    """List S3 objects under a prefix, returning (key, filename) pairs.

    Args:
        s3: boto3 S3 client.
        prefix: S3 key prefix to scan.
        suffix: Optional suffix filter (e.g. ".parquet", ".zip").

    Returns:
        List of (full_key, filename) tuples.
    """
    results: list[tuple[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.rsplit("/", 1)[-1]
            if suffix and not fname.endswith(suffix):
                continue
            if fname:
                results.append((key, fname))
    return results


def get_completed_counties(s3: Any) -> set[str]:
    """Scan S3 for already-completed county parquets.

    Args:
        s3: boto3 S3 client.

    Returns:
        Set of completed FIPS codes.
    """
    completed: set[str] = set()
    for _key, fname in _s3_list_prefix(
        s3, f"{DATA_PREFIX}/flood_county_areas/", suffix=".parquet"
    ):
        fips = fname.replace(".parquet", "")
        completed.add(fips)
    return completed


def get_conus_counties(s3: Any) -> dict[str, dict[str, str]]:
    """List raw FEMA zips, dedup by FIPS keeping latest date, CONUS only.

    Args:
        s3: boto3 S3 client.

    Returns:
        Dict mapping FIPS -> {"key", "dfirm_id", "date", "fips"}.
    """
    raw_files = _s3_list_prefix(
        s3, f"{DATA_PREFIX}/flood_raw/", suffix=".zip"
    )

    by_fips: dict[str, dict[str, str]] = {}
    for key, fname in raw_files:
        dfirm_id = fname.split("_")[0] if "_" in fname else fname[:-4]
        fips = dfirm_id[:5]

        if fips[:2] in NON_CONUS_PREFIXES:
            continue

        m = re.search(r"_(\d{8})\.", fname)
        entry_date = m.group(1) if m else "00000000"

        if fips not in by_fips or entry_date > by_fips[fips]["date"]:
            by_fips[fips] = {
                "key": key,
                "dfirm_id": dfirm_id,
                "date": entry_date,
                "fips": fips,
            }

    return by_fips


def split_into_shards(
    counties: list[dict[str, str]],
    n_shards: int,
) -> list[list[dict[str, str]]]:
    """Split county list into N roughly equal shards.

    Args:
        counties: List of county dicts with fips/key/dfirm_id.
        n_shards: Number of shards to create.

    Returns:
        List of N lists, each containing a subset of counties.
    """
    shard_size = math.ceil(len(counties) / n_shards)
    shards: list[list[dict[str, str]]] = []
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, len(counties))
        if start < len(counties):
            shards.append(counties[start:end])
    return shards


def deploy_code(s3: Any) -> None:
    """Upload code files to S3.

    Args:
        s3: boto3 S3 client.
    """
    for fname in CODE_FILES:
        local = CODE_DIR / fname
        if not local.exists():
            print(f"  ERROR: {local} not found")
            sys.exit(1)
        key = f"{CODE_PREFIX}/{fname}"
        s3.upload_file(str(local), BUCKET, key)
        print(f"  Uploaded: s3://{BUCKET}/{key}")


def upload_county_list(
    s3: Any,
    shard_idx: int,
    counties: list[dict[str, str]],
) -> str:
    """Upload a shard's county list JSON to S3.

    Args:
        s3: boto3 S3 client.
        shard_idx: Shard index (0-based).
        counties: List of county dicts for this shard.

    Returns:
        S3 key where the JSON was uploaded.
    """
    payload = [
        {"fips": c["fips"], "key": c["key"], "dfirm_id": c["dfirm_id"]}
        for c in counties
    ]
    key = f"{CODE_PREFIX}/county_list_shard_{shard_idx}.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"  Uploaded: s3://{BUCKET}/{key} ({len(payload)} counties)")
    return key


def launch_shard(
    sm: Any,
    s3_county_key: str,
    shard_idx: int,
    timestamp: str,
    instance_type: str,
    role_arn: str,
) -> str:
    """Launch a SageMaker processing job for one shard.

    Args:
        sm: boto3 SageMaker client.
        s3_county_key: S3 key for this shard's county list JSON.
        shard_idx: Shard index (0-based).
        timestamp: Timestamp string for job naming.
        instance_type: EC2 instance type.
        role_arn: SageMaker execution role ARN.

    Returns:
        Job name of the launched job.
    """
    job_name = f"geocert-flood-shard-{shard_idx}-{timestamp}"
    image_uri = get_image_uri(REGION)

    config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 50,
            },
        },
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "bash",
                "/opt/ml/processing/input/code/entrypoint_flood_shard.sh",
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
                "InputName": "tiger",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{TIGER_PREFIX}/",
                    "LocalPath": "/opt/ml/processing/input/tiger",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "crosswalk",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{DATA_PREFIX}/zcta_county_crosswalk.parquet",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "county_list",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{s3_county_key}",
                    "LocalPath": "/opt/ml/processing/input/county_list",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 21600},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
        },
    }

    response = sm.create_processing_job(**config)
    print(f"  Launched: {response['ProcessingJobArn']}")
    return job_name


def main() -> None:
    """Orchestrate the sharded flood pipeline."""
    parser = argparse.ArgumentParser(
        description="Launch sharded flood pipeline on SageMaker"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without launching")
    parser.add_argument("--n-shards", type=int, default=4,
                        help="Number of parallel shards (default: 4)")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge",
                        help="Instance type per shard (default: ml.m5.4xlarge)")
    parser.add_argument("--force", action="store_true",
                        help="Launch even if no remaining counties")
    parser.add_argument("--fips-min", default=None,
                        help="Only process counties with FIPS >= this value (e.g. 29000)")
    parser.add_argument("--fips-max", default=None,
                        help="Only process counties with FIPS <= this value")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    role_arn = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
    s3 = boto3.client("s3", **_aws)

    # --- Scan S3 for state ---
    print("=== SCANNING S3 ===")
    print("  Listing completed counties...")
    completed = get_completed_counties(s3)
    print(f"  Found {len(completed)} completed counties")

    print("  Listing raw FEMA zips (dedup by FIPS, CONUS only)...")
    all_counties = get_conus_counties(s3)
    print(f"  Found {len(all_counties)} unique CONUS counties")

    # --- Subtract completed + apply FIPS range filter ---
    remaining = []
    for fips, v in sorted(all_counties.items()):
        if fips in completed:
            continue
        if args.fips_min and fips < args.fips_min:
            continue
        if args.fips_max and fips > args.fips_max:
            continue
        remaining.append(v)

    # --- Compute shard sizes ---
    n_shards = min(args.n_shards, len(remaining)) if remaining else 0
    approx_per_shard = math.ceil(len(remaining) / n_shards) if n_shards else 0

    # --- Summary ---
    print("")
    print("=== SHARDED FLOOD PIPELINE ===")
    print(f"Total CONUS counties: {len(all_counties)}")
    print(f"Already completed:    {len(completed)}")
    print(f"Remaining:           {len(remaining)}")
    if n_shards > 0:
        print(f"Shards:              {n_shards} x ~{approx_per_shard} counties")
    else:
        print("Shards:              0 (nothing to do)")
    print(f"Instance:            {args.instance_type} per shard")
    if args.fips_min or args.fips_max:
        print(f"FIPS range:          {args.fips_min or '*'} to {args.fips_max or '*'}")
    if remaining:
        print(f"FIPS span:           {remaining[0]['fips']} .. {remaining[-1]['fips']}")

    if not remaining and not args.force:
        print("\nAll counties completed. Use --force to launch anyway.")
        return

    if not remaining and args.force:
        print("\nWARNING: No remaining counties but --force specified.")
        print("Nothing to shard. Exiting.")
        return

    # --- Split into shards ---
    shards = split_into_shards(remaining, n_shards)

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(shards)} shards:")
        for i, shard in enumerate(shards):
            print(f"  Shard {i}: {len(shard)} counties "
                  f"(FIPS {shard[0]['fips']}..{shard[-1]['fips']})")
        return

    # --- Deploy code ---
    print("\n=== DEPLOYING CODE ===")
    deploy_code(s3)

    # --- Upload county lists and launch jobs ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_names: list[str] = []

    print("\n=== UPLOADING COUNTY LISTS ===")
    county_keys: list[str] = []
    for i, shard in enumerate(shards):
        key = upload_county_list(s3, i, shard)
        county_keys.append(key)

    print("\n=== LAUNCHING JOBS ===")
    sm = boto3.client("sagemaker", **_aws)
    for i, key in enumerate(county_keys):
        print(f"\n  Shard {i} ({len(shards[i])} counties):")
        name = launch_shard(sm, key, i, timestamp, args.instance_type, role_arn)
        job_names.append(name)

    # --- Print monitor commands ---
    print("\n" + "=" * 60)
    print("MONITOR COMMANDS")
    print("=" * 60)
    for name in job_names:
        print(f"\n--- {name} ---")
        print(f"  MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
              f"--processing-job-name {name} --region {REGION} "
              f"--profile nsc-swarm --query 'ProcessingJobStatus'")
        print(f"  MSYS_NO_PATHCONV=1 aws logs tail "
              f"/aws/sagemaker/ProcessingJobs "
              f"--log-stream-name-prefix {name} --follow "
              f"--profile nsc-swarm --region {REGION}")


if __name__ == "__main__":
    main()
