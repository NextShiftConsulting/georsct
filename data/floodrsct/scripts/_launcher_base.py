"""
_launcher_base.py -- Shared SageMaker launcher utilities for s035.

All s035 launchers import from here. Do not run directly.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REGION = "us-east-1"
ACCOUNT_ID = "865679935554"
SAGEMAKER_ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
DATA_BUCKET = "swarm-floodrsct-data"
CODE_BUCKET = "swarm-floodrsct-data"

# SageMaker-managed PyTorch CPU image (no GPU needed for data pulls)
PYTORCH_CPU = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
)
# For jobs that need GPU (surrogate training)
PYTORCH_GPU = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.5.1-gpu-py311-cu121-ubuntu20.04-sagemaker"
)

SERIES_DIR = Path(__file__).parent.parent
JOBS_DIR = SERIES_DIR / "jobs"
CONFIGS_DIR = SERIES_DIR / "configs"


def make_job_name(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"s035-{prefix}-{ts}"


def upload_code(job_name: str, job_script: str, extra_files: list[str] | None = None) -> str:
    """Upload job script + _manifest_writer + configs as individual files to S3.

    Files land at s3://{CODE_BUCKET}/code/s035/{job_name}/src/*.
    SageMaker S3Prefix input downloads the whole prefix so every file is
    directly available in /opt/ml/processing/input/code/ — no unzip needed.
    """
    s3 = boto3.client("s3", **get_aws_credentials())

    # Core files: the job script + ALL _*.py shared helpers in jobs/
    files_to_upload: list[Path] = [JOBS_DIR / job_script]
    for helper in sorted(JOBS_DIR.glob("_*.py")):
        files_to_upload.append(helper)
    # Vendored wheels (e.g. swarm_auth)
    for whl in sorted(JOBS_DIR.glob("*.whl")):
        files_to_upload.append(whl)
    if extra_files:
        for f in extra_files:
            p = Path(f)
            if p.exists():
                files_to_upload.append(p)
            else:
                log.warning("Extra file not found, skipping: %s", f)

    # Always include all configs
    for cfg in CONFIGS_DIR.glob("*.yaml"):
        files_to_upload.append(cfg)

    prefix = f"code/s035/{job_name}/src"
    for fp in files_to_upload:
        key = f"{prefix}/{fp.name}"
        s3.upload_file(str(fp), CODE_BUCKET, key)

    s3_prefix = f"{prefix}/"
    log.info("Uploaded code to s3://%s/%s (%d files)", CODE_BUCKET, s3_prefix, len(files_to_upload))
    return s3_prefix


def _upload_bootstrap(code_prefix: str, script_content: str) -> None:
    """Upload a bootstrap shell script alongside the code."""
    s3 = boto3.client("s3", **get_aws_credentials())
    key = f"{code_prefix}_bootstrap.sh"
    s3.put_object(Bucket=CODE_BUCKET, Key=key, Body=script_content.encode())


# Default packages installed in every job container.
# Extend per-job via the pip_packages argument — do NOT add geopandas/rasterio
# here; only spatial jobs need them, and they add 2-3 min install time on the
# PyTorch image which lacks native GDAL.
_BASE_PACKAGES = "pyyaml pyarrow pandas requests boto3"


def launch_processing_job(
    job_name: str,
    job_script: str,
    job_args: list[str],
    instance_type: str = "ml.m5.large",
    extra_files: list[str] | None = None,
    volume_size_gb: int = 50,
    pip_packages: str | None = None,
    image_uri: str | None = None,
    env_overrides: dict[str, str] | None = None,
    pre_install_cmd: str | None = None,
    dry_run: bool = False,
) -> str:
    """Upload code and launch a SageMaker Processing job.

    Args:
        pip_packages: Space-separated packages to install *in addition to*
            _BASE_PACKAGES.  Pass e.g. ``"geopandas rasterio"`` for spatial
            jobs.  None means base packages only.
        image_uri: Override the default PYTORCH_CPU image. Pass PYTORCH_GPU
            for jobs that require a CUDA forward pass (e.g. Prithvi smoke test).
        env_overrides: Additional environment variables to inject into the
            container (e.g. {"EARTHDATA_TOKEN": token}).  Merged on top of
            the default environment; caller-supplied values take precedence.
        pre_install_cmd: Shell command to run before pip install (e.g. apt-get
            for system-level dependencies like libgdal-dev).
    """
    code_prefix = upload_code(job_name, job_script, extra_files)

    packages = _BASE_PACKAGES
    if pip_packages:
        packages = f"{_BASE_PACKAGES} {pip_packages}"

    processing_inputs = [
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri": f"s3://{CODE_BUCKET}/{code_prefix}",
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
    ]

    # Jobs upload directly to S3 via boto3 during execution; no output mount needed.
    processing_outputs: list[dict] = []

    args_str = " ".join(job_args)

    # SageMaker caps each ContainerEntrypoint member at 256 chars.
    # Write a bootstrap script to avoid the limit.
    pre_cmd = f"{pre_install_cmd}\n" if pre_install_cmd else ""
    bootstrap = (
        "#!/bin/bash\nset -e\n"
        f"{pre_cmd}"
        f"pip install -qU {packages}\n"
        "pip install -q /opt/ml/processing/input/code/*.whl 2>/dev/null || true\n"
        "cd /opt/ml/processing/input/code\n"
        f"python -u {job_script} {args_str}\n"
    )
    _upload_bootstrap(code_prefix, bootstrap)

    container_cmd = [
        "bash",
        f"/opt/ml/processing/input/code/_bootstrap.sh",
    ]

    job_config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": volume_size_gb,
            }
        },
        "AppSpecification": {
            "ImageUri": image_uri or PYTORCH_CPU,
            "ContainerEntrypoint": container_cmd,
        },
        "ProcessingInputs": processing_inputs,
        **( {"ProcessingOutputConfig": {"Outputs": processing_outputs}} if processing_outputs else {} ),
        "RoleArn": SAGEMAKER_ROLE,
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "AWS_DEFAULT_REGION": REGION,
            # Route all HF Hub + Datasets cache to EBS (/tmp), not root fs (~20 GB limit)
            "HF_HOME": "/tmp/hf_cache",
            "HF_DATASETS_CACHE": "/tmp/hf_cache/datasets",
            **(env_overrides or {}),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
    }

    if dry_run:
        log.info("[DRY RUN] Would launch job: %s", job_name)
        log.info("[DRY RUN] Config:\n%s", json.dumps(job_config, indent=2))
        return job_name

    sm = boto3.client("sagemaker", **get_aws_credentials())
    sm.create_processing_job(**job_config)
    log.info("Launched SageMaker job: %s (instance: %s)", job_name, instance_type)
    log.info(
        "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix %s --follow",
        job_name,
    )
    return job_name


def wait_for_job(job_name: str, poll_interval: int = 30) -> str:
    """Poll until job completes or fails. Returns final status."""
    sm = boto3.client("sagemaker", **get_aws_credentials())
    log.info("Waiting for job: %s", job_name)
    while True:
        resp = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            log.info("Job %s finished with status: %s", job_name, status)
            if status == "Failed":
                log.error("Failure reason: %s", resp.get("FailureReason", "unknown"))
            return status
        log.info("  %s: %s", job_name, status)
        time.sleep(poll_interval)
