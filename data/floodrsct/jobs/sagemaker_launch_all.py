#!/usr/bin/env python3
"""SageMaker launcher for Data Lock A fetch jobs (s035).

LOCAL script -- runs on the user's machine, uploads code to S3, and
launches SageMaker Processing Jobs for each data fetcher.

Usage:
    python sagemaker_launch_all.py --job mrms --event harvey2017
    python sagemaker_launch_all.py --job mrms --all
    python sagemaker_launch_all.py --job all
    python sagemaker_launch_all.py --job nlcd --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from swarm_auth import get_aws_credentials

import boto3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::865679935554:role/SageMakerExecutionRole"
IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5-cpu-py311"
)
CODE_BUCKET = "swarm-floodrsct-data"
MAX_RUNTIME_SECONDS = 43200  # 12 hours

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared utilities uploaded with every job
SHARED_FILES = ["_manifest_writer.py", "_s3_stream.py"]

JOB_CONFIG = {
    "mrms": {
        "script": "fetch_noaa_mrms_v2.py",
        "entrypoint": "entrypoint_mrms.sh",
        "instance": "ml.m5.xlarge",
        "volume_gb": 30,
        "events": [
            "harvey2017",
            "imelda2019",
            "beryl2024",
            "ida2021_nyc",
            "ian2022",
            "hilary2023",
        ],
    },
    "tides": {
        "script": "fetch_noaa_tides.py",
        "entrypoint": "entrypoint_tides.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": [
            "harvey2017",
            "imelda2019",
            "beryl2024",
            "ida2021_nyc",
            "ian2022",
            "hilary2023",
        ],
    },
    "surge": {
        "script": "fetch_surge_hwm.py",
        "entrypoint": "entrypoint_surge.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": [
            "harvey2017",
            "imelda2019",
            "beryl2024",
            "ida2021_nyc",
            "ian2022",
            "hilary2023",
        ],
    },
    "nlcd": {
        "script": "fetch_nlcd_impervious.py",
        "entrypoint": "entrypoint_nlcd.sh",
        "instance": "ml.m5.xlarge",
        "volume_gb": 100,
        "events": None,
    },
    "dem": {
        "script": "fetch_3dep_dem.py",
        "entrypoint": "entrypoint_dem.sh",
        "instance": "ml.m5.2xlarge",
        "volume_gb": 50,
        "events": [
            "houston",
            "southwest_florida",
            "nyc",
            "socal",
            "new_orleans",
        ],
    },
    "hurdat2": {
        "script": "fetch_hurdat2.py",
        "entrypoint": "entrypoint_hurdat2.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "openfema": {
        "script": "fetch_openfema_event.py",
        "entrypoint": "entrypoint_openfema.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "geocertdb": {
        "script": "copy_geocertdb2026.py",
        "entrypoint": "entrypoint_geocertdb.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "sewersheds": {
        "script": "fetch_nyc_sewersheds.py",
        "entrypoint": "entrypoint_sewersheds.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "levees": {
        "script": "fetch_usace_levees.py",
        "entrypoint": "entrypoint_levees.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "nyc311": {
        "script": "fetch_nyc_311.py",
        "entrypoint": "entrypoint_nyc311.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "houston311": {
        "script": "fetch_houston_311.py",
        "entrypoint": "entrypoint_houston311.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
    "slosh": {
        "script": "fetch_noaa_slosh.py",
        "entrypoint": "entrypoint_slosh.sh",
        "instance": "ml.m5.large",
        "volume_gb": 10,
        "events": None,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_job_name(job_type: str, event: str | None) -> str:
    """Generate a unique job name: s035-fetch-{job}-{event}-{ts}.

    SageMaker requires: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
    So replace underscores with hyphens.
    """
    parts = ["s035", "fetch", job_type]
    if event:
        parts.append(event.replace("_", "-"))
    parts.append(TIMESTAMP)
    return "-".join(parts)


def _upload_code(s3, job_name: str, cfg: dict) -> str:
    """Upload job script, entrypoint, and shared files to S3.

    Returns:
        The S3 prefix where code was uploaded.
    """
    code_prefix = f"code/s035/{job_name}/src"
    files_to_upload = [cfg["script"], cfg["entrypoint"]] + SHARED_FILES

    for filename in files_to_upload:
        local_path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Required file not found: {local_path}")
        s3_key = f"{code_prefix}/{filename}"
        s3.upload_file(local_path, CODE_BUCKET, s3_key)
        print(f"  Uploaded s3://{CODE_BUCKET}/{s3_key}")

    return code_prefix


def _launch_job(
    sm,
    s3,
    job_type: str,
    event: str | None,
    cfg: dict,
    dry_run: bool,
) -> dict:
    """Upload code and create a single SageMaker Processing Job.

    Returns:
        dict with job_name, job_type, event, status.
    """
    job_name = _make_job_name(job_type, event)
    result = {
        "job_name": job_name,
        "job_type": job_type,
        "event": event or "-",
        "instance": cfg["instance"],
        "status": "PENDING",
    }

    print(f"\n{'=' * 64}")
    print(f"Job:      {job_type}")
    print(f"Event:    {event or '(none)'}")
    print(f"Name:     {job_name}")
    print(f"Instance: {cfg['instance']}")
    print(f"Volume:   {cfg['volume_gb']} GB")
    print(f"{'=' * 64}")

    if dry_run:
        result["status"] = "DRY_RUN"
        print("  [DRY RUN] Skipping launch.")
        return result

    # Upload code
    code_prefix = _upload_code(s3, job_name, cfg)
    code_s3_uri = f"s3://{CODE_BUCKET}/{code_prefix}"

    # Container arguments -- omit key entirely if no event (SageMaker rejects empty list)
    app_spec = {
        "ImageUri": IMAGE_URI,
        "ContainerEntrypoint": [
            "bash",
            f"/opt/ml/processing/input/code/{cfg['entrypoint']}",
        ],
    }
    if event:
        app_spec["ContainerArguments"] = [event]

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": cfg["instance"],
                "VolumeSizeInGB": cfg["volume_gb"],
            },
        },
        AppSpecification=app_spec,
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        RoleArn=ROLE_ARN,
        Environment={"PYTHONUNBUFFERED": "1"},
        StoppingCondition={"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        Tags=[
            {"Key": "project", "Value": "rsct-geocert"},
            {"Key": "pipeline", "Value": "data-lock-a"},
            {"Key": "job_type", "Value": job_type},
        ],
    )

    result["status"] = "LAUNCHED"
    print(f"  LAUNCHED: {job_name}")
    print(f"\n  Monitor:")
    print(
        f"    MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job "
        f"--processing-job-name {job_name} --region {REGION} "
        f"--query ProcessingJobStatus --output text"
    )
    print(
        f"    MSYS_NO_PATHCONV=1 aws logs tail "
        f"/aws/sagemaker/ProcessingJobs --log-stream-name-prefix {job_name} "
        f"--follow --region {REGION}"
    )
    return result


def _collect_launches(
    job_type: str,
    event: str | None,
    all_events: bool,
) -> list[tuple[str, str | None]]:
    """Resolve CLI args into a list of (job_type, event) pairs to launch."""
    launches: list[tuple[str, str | None]] = []

    if job_type == "all":
        for jt, cfg in JOB_CONFIG.items():
            if cfg["events"]:
                for ev in cfg["events"]:
                    launches.append((jt, ev))
            else:
                launches.append((jt, None))
        return launches

    cfg = JOB_CONFIG[job_type]

    if cfg["events"] is None:
        # Non-event job (nlcd, hurdat2)
        launches.append((job_type, None))
    elif all_events:
        for ev in cfg["events"]:
            launches.append((job_type, ev))
    elif event:
        if event not in cfg["events"]:
            print(
                f"ERROR: event '{event}' not in {job_type} events: "
                f"{cfg['events']}"
            )
            sys.exit(1)
        launches.append((job_type, event))
    else:
        print(f"ERROR: --event or --all required for job '{job_type}'")
        sys.exit(1)

    return launches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Data Lock A fetch jobs on SageMaker"
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=list(JOB_CONFIG.keys()) + ["all"],
        help="Job type to launch",
    )
    parser.add_argument("--event", default=None, help="Storm event ID")
    parser.add_argument(
        "--all",
        dest="all_events",
        action="store_true",
        help="Launch for all events",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without launching",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("Data Lock A -- SageMaker Launcher (s035)")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Dry run:   {args.dry_run}")
    print("=" * 64)

    # Resolve launches
    launches = _collect_launches(args.job, args.event, args.all_events)
    print(f"\nJobs to launch: {len(launches)}")
    for jt, ev in launches:
        print(f"  {jt:10s}  {ev or '(global)'}")

    # Validate all files exist before launching anything
    files_needed: set[str] = set(SHARED_FILES)
    for jt, _ in launches:
        cfg = JOB_CONFIG[jt]
        files_needed.add(cfg["script"])
        files_needed.add(cfg["entrypoint"])

    missing = [
        f for f in files_needed
        if not os.path.isfile(os.path.join(SCRIPT_DIR, f))
    ]
    if missing:
        print(f"\nFATAL: Missing files: {missing}")
        sys.exit(1)
    print("\nAll code files validated.")

    # AWS clients
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)  # avoid duplicate kwarg
    sm = boto3.client("sagemaker", region_name=REGION, **_aws)
    s3 = boto3.client("s3", region_name=REGION, **_aws)

    # Launch
    results: list[dict] = []
    for jt, ev in launches:
        cfg = JOB_CONFIG[jt]
        try:
            r = _launch_job(sm, s3, jt, ev, cfg, dry_run=args.dry_run)
        except Exception as exc:
            r = {
                "job_name": _make_job_name(jt, ev),
                "job_type": jt,
                "event": ev or "-",
                "instance": cfg["instance"],
                "status": f"FAILED: {exc}",
            }
            print(f"  FAILED: {exc}")
        results.append(r)

    # Summary table
    print(f"\n{'=' * 64}")
    print("LAUNCH SUMMARY")
    print(f"{'=' * 64}")
    print(f"{'Job':<10s} {'Event':<16s} {'Instance':<14s} {'Status':<10s} {'Name'}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['job_type']:<10s} {r['event']:<16s} {r['instance']:<14s} "
            f"{r['status']:<10s} {r['job_name']}"
        )

    launched = sum(1 for r in results if r["status"] == "LAUNCHED")
    failed = sum(1 for r in results if r["status"].startswith("FAILED"))
    print(f"\nTotal: {len(results)} | Launched: {launched} | Failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
