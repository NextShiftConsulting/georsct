"""
_launcher_base.py -- Shared SageMaker launcher utilities for s035.

All s035 launchers import from here. Do not run directly.
"""

import json
import logging
import subprocess
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

# Ecosystem wheels (yrsn, rsct, etc.) built by preflight_wheels.py in
# yrsn-experiments and uploaded to this prefix.  Mounted as a separate
# SageMaker ProcessingInput -- matches the s018/s019 pattern.
WHEELS_BUCKET = "swarm-yrsn-datasets"
WHEELS_PREFIX = "rsct_code/wheels/20260610-070717"

# SageMaker-managed scikit-learn image -- DO NOT USE AS DEFAULT.
# sklearn images (683313688378) ship Python 3.9 with ancient pinned deps
# (pandas 1.1.3, numpy <1.24). Upgrading any dep breaks numpy C extensions.
# Kept for reference only; use PYTORCH_CPU for all data extraction jobs.
SKLEARN_CPU = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)
# SageMaker-managed PyTorch CPU image (for jobs that import torch)
PYTORCH_CPU = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
)
# For jobs that need GPU (surrogate training)
PYTORCH_GPU = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.5.1-gpu-py311-cu121-ubuntu20.04-sagemaker"
)
# Default image for data extraction / feature engineering jobs.
# PyTorch CPU is the only viable base -- sklearn image ships Python 3.9
# with pinned numpy that breaks when any dep is upgraded.
DEFAULT_IMAGE = PYTORCH_CPU

SERIES_DIR = Path(__file__).parent.parent
JOBS_DIR = SERIES_DIR / "jobs"
CONFIGS_DIR = SERIES_DIR / "configs"

# Approved SageMaker instance types.  Prevents accidental expensive or
# unsupported instance launches.  Extend here when GPU jobs are needed.
ALLOWED_INSTANCES = frozenset({
    "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge",
    "ml.m5.4xlarge", "ml.m5.8xlarge",
    "ml.m7i.8xlarge",
})


def _get_git_info() -> dict[str, str]:
    """Capture git commit hash and dirty status for traceability."""
    info = {"git_hash": "unknown", "git_dirty": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(SERIES_DIR),
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=str(SERIES_DIR),
        )
        if result.returncode == 0:
            info["git_dirty"] = "true" if result.stdout.strip() else "false"
    except Exception:
        pass
    return info


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
            if not p.exists():
                # Try relative to JOBS_DIR
                p = JOBS_DIR / f
            if p.exists():
                files_to_upload.append(p)
            else:
                log.warning("Extra file not found, skipping: %s", f)

    # Always include all configs
    for cfg in CONFIGS_DIR.glob("*.yaml"):
        files_to_upload.append(cfg)

    # FEATURE_CONTRACT.yaml lives one level above jobs/ — include it so
    # _validate_contract.py can find it on SageMaker.
    contract = SERIES_DIR / "FEATURE_CONTRACT.yaml"
    if contract.exists():
        files_to_upload.append(contract)

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


def _run_preflight(phase_id: str, scenario: str | None = None) -> bool:
    """Run experiment contract preflight. Returns True if no FAILs."""
    sys.path.insert(0, str(JOBS_DIR))
    try:
        from validate_experiment_readiness import preflight
        _, fails, _ = preflight(phase_id, scenario)
        return fails == 0
    except ImportError:
        log.warning("validate_experiment_readiness.py not found; skipping preflight")
        return True
    finally:
        if str(JOBS_DIR) in sys.path:
            sys.path.remove(str(JOBS_DIR))


# Instance metadata: vCPUs and RAM (GB) for resource audit.
_INSTANCE_SPECS = {
    "ml.m5.large":    (2,   8),
    "ml.m5.xlarge":   (4,  16),
    "ml.m5.2xlarge":  (8,  32),
    "ml.m5.4xlarge": (16,  64),
    "ml.m5.8xlarge": (32, 128),
    "ml.m7i.8xlarge": (32, 128),
}


def _resource_audit(
    job_script: str,
    instance_type: str,
    volume_size_gb: int,
    pip_packages: str | None,
    image_uri: str | None,
    timeout_s: int,
) -> tuple[list[str], list[str]]:
    """9-dimension resource audit. Returns (warnings, blockers).

    Blockers are hard stops -- the launch MUST NOT proceed.
    Warnings are informational -- review recommended but not blocking.

    Dimensions checked:
      1. Memory   -- data size vs instance RAM
      2. Cache    -- (informational only)
      3. Thread   -- parallelism keywords vs instance vCPUs (BLOCKS if serial on multi-core)
      4. Image    -- which base image
      5. Instance -- type and specs
      6. Volume   -- EBS sizing
      7. Pip      -- packages requested
      8. Pre-install -- (checked at call site)
      9. Timeout  -- max runtime
    """
    warnings: list[str] = []
    blockers: list[str] = []
    vcpus, ram_gb = _INSTANCE_SPECS.get(instance_type, (0, 0))

    # --- Dim 3: Thread audit (HARD BLOCK) ---
    script_path = JOBS_DIR / job_script
    if script_path.exists():
        code = script_path.read_text(encoding="utf-8", errors="replace")
        has_parallel = any(kw in code for kw in [
            "n_jobs", "Parallel(", "Pool(", "ThreadPool(",
            "multiprocessing", "concurrent.futures",
        ])
        if not has_parallel and vcpus > 1:
            blockers.append(
                "THREAD: No parallelism detected in %s but instance "
                "has %d vCPUs -- %d cores will be idle. "
                "Add ProcessPoolExecutor, joblib.Parallel, or n_jobs=-1. "
                "LAUNCH BLOCKED until parallelism is added or instance "
                "is downgraded to ml.m5.large (2 vCPU)." % (job_script, vcpus, vcpus - 1)
            )
        if has_parallel and vcpus <= 1:
            warnings.append(
                "THREAD: Parallelism in %s but instance %s has only "
                "%d vCPU -- consider a larger instance." % (job_script, instance_type, vcpus)
            )
    else:
        warnings.append("THREAD: Job script %s not found locally for audit." % job_script)

    # --- Dim 1: Memory budget (HARD BLOCK for known workloads) ---
    # For jobs with ProcessPoolExecutor, compute actual per-worker memory
    # from array allocations and cross-check against instance RAM.
    if script_path.exists() and ram_gb > 0:
        code = script_path.read_text(encoding="utf-8", errors="replace")
        # Extract worker count from min(N, ...) pattern
        import re as _re
        worker_match = _re.search(r'n_workers\s*=\s*min\(\s*(\d+)', code)
        n_workers = int(worker_match.group(1)) if worker_match else 1

        # Estimate per-worker memory for raster jobs processing 10812x10812 tiles
        # Each float64 array at 10812x10812 = ~895 MB
        if "10812" in code or "compute_flow_accumulation" in code or "compute_hand" in code:
            # Known arrays per worker (peak during D8 flow direction):
            #   DEM (float64): 895 MB
            #   padded DEM (float64, +2 border): 896 MB
            #   drops (8 x rows x cols float64): 8 x 895 = 7160 MB  <-- dominates
            #   fdir (int8): 112 MB
            #   flow_acc (float64): 895 MB
            #   in_degree (int32): 448 MB
            #   4 metric arrays (float64): 4 x 895 = 3580 MB
            #   _trace_downstream_vectorized: 5 arrays (int32): 5 x 448 = 2240 MB
            #   GFI calls trace again: +2240 MB
            #   DEM conditioning (pit fill + flat resolution):
            #     DEM copy (895) + seed (895) + dist_transform (895) = 2685 MB
            #   Python/numpy/GC overhead: ~1000 MB
            # D8 drops may not be GC'd before next phase starts in same worker.
            per_worker_gb = (895 + 896 + 7160 + 112 + 895 + 448
                             + 3580 + 2240 + 2240 + 2685 + 1000) / 1024
            total_gb = per_worker_gb * n_workers
            headroom = ram_gb * 0.70  # 30% reserved for OS + Python parent

            if total_gb > headroom:
                blockers.append(
                    "MEMORY: %d workers x %.1f GB/worker = %.0f GB > %.0f GB "
                    "usable (70%% of %d GB). Reduce n_workers to %d or use a "
                    "larger instance. LAUNCH BLOCKED."
                    % (n_workers, per_worker_gb, total_gb, headroom,
                       ram_gb, int(headroom / per_worker_gb))
                )
            else:
                warnings.append(
                    "MEMORY: %d workers x %.1f GB/worker = %.0f GB of %.0f GB "
                    "usable (%.0f%% utilization)."
                    % (n_workers, per_worker_gb, total_gb, headroom,
                       total_gb / headroom * 100)
                )
        elif ram_gb > 32:
            warnings.append(
                "MEMORY: Instance %s has %d GB RAM. Most s035 jobs need "
                "<8 GB. Consider a smaller instance." % (instance_type, ram_gb)
            )

    # --- Dim 5: Instance summary ---
    if vcpus == 0:
        warnings.append("INSTANCE: Unknown instance type %s -- cannot audit specs." % instance_type)

    # --- Dim 6: Volume ---
    if volume_size_gb > 30:
        warnings.append(
            "VOLUME: %d GB requested. Most s035 jobs need <10 GB. "
            "Oversized volumes waste startup time." % volume_size_gb
        )

    # --- Dim 7b: Wheel dependency cross-check (HARD BLOCK) ---
    # Ecosystem wheels are installed with --no-deps. If the job script
    # imports from a wheel that needs packages not in pip_packages and not
    # in the base image, those imports will fail at runtime.
    # Check floodcaster specifically (root cause of 2026-06-30 STAC gap
    # AND 2026-07-01 planetary_computer import crash on all 5 scenarios).
    if pip_packages and script_path.exists():
        code = script_path.read_text(encoding="utf-8", errors="replace")
        if "floodcaster" in code:
            # These are floodcaster's runtime deps that are NOT in the
            # PyTorch base image. Each must appear in pip_packages.
            # Even if the job only uses floodcaster.hydrology, the wheel's
            # __init__.py imports these transitively.
            _FC_REQUIRED_DEPS = {
                "pystac-client": "pystac_client",   # STAC access
                "planetary-computer": "planetary_computer",  # PC token signing
                "rasterio": "rasterio",              # raster I/O
                "geopandas": "geopandas",            # spatial joins
            }
            pip_set = set(pip_packages.split())
            for pip_name, import_name in _FC_REQUIRED_DEPS.items():
                if pip_name not in pip_set:
                    blockers.append(
                        "PIP: Job imports floodcaster but pip_packages is "
                        "missing '%s'. Ecosystem wheels are installed with "
                        "--no-deps, so this WILL fail at runtime. "
                        "LAUNCH BLOCKED until dep is added."
                        % pip_name
                    )

    # --- Dim 7c: Import smoke test (HARD BLOCK) ---
    # Actually try importing the job's key packages locally.
    # Catches transitive import failures that static pip_packages checks miss
    # (e.g. floodcaster.__init__.py importing planetary_computer even when
    # only floodcaster.hydrology is used by the job).
    if script_path.exists():
        code = script_path.read_text(encoding="utf-8", errors="replace")
        # Extract "from X import ..." and "import X" statements
        import re
        import_lines = re.findall(
            r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
            code, re.MULTILINE,
        )
        modules_to_test = set()
        for from_mod, import_mod in import_lines:
            mod = from_mod or import_mod
            # Only test ecosystem/project packages, not stdlib
            top = mod.split(".")[0]
            if top in ("floodcaster", "georsct", "rsct", "yrsn"):
                modules_to_test.add(mod)
        if modules_to_test:
            import subprocess as _sp
            for mod in sorted(modules_to_test):
                try:
                    result = _sp.run(
                        [sys.executable, "-c", f"import {mod}"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode != 0:
                        blockers.append(
                            "IMPORT: 'import %s' failed locally. "
                            "This WILL fail on SageMaker too. "
                            "Fix the import or add missing deps. "
                            "Error: %s" % (mod, result.stderr.strip()[:200])
                        )
                except Exception as e:
                    warnings.append(
                        "IMPORT: Could not verify 'import %s': %s" % (mod, e)
                    )

    # --- Dim 9: Timeout ---
    if timeout_s > 14400:
        warnings.append(
            "TIMEOUT: %ds (%.1fh) is very long. Typical s035 jobs "
            "run <1h." % (timeout_s, timeout_s / 3600)
        )

    # --- Print audit summary ---
    image_label = "pytorch-cpu" if (image_uri or DEFAULT_IMAGE) == PYTORCH_CPU else (
        "pytorch-gpu" if (image_uri or "") == PYTORCH_GPU else "custom"
    )
    print()
    print("=" * 64)
    print("RESOURCE AUDIT (9 dimensions)")
    print("=" * 64)
    print("  1. Memory     : %s (%d vCPU, %d GB RAM)" % (instance_type, vcpus, ram_gb))
    print("  2. Cache      : (no shared cache -- per-job isolation)")
    print("  3. Thread     : %s" % (
        "parallelism detected" if (script_path.exists() and any(
            kw in script_path.read_text(encoding="utf-8", errors="replace")
            for kw in ["n_jobs", "Parallel(", "Pool(", "ThreadPool(",
                       "multiprocessing", "concurrent.futures"]
        )) else "SERIAL (single-threaded) ** WILL BLOCK LAUNCH **"
    ))
    print("  4. Image      : %s" % image_label)
    print("  5. Instance   : %s" % instance_type)
    print("  6. Volume     : %d GB" % volume_size_gb)
    print("  7a. Pip       : %s" % (pip_packages or "(base only)"))
    print("  7b. Wheel deps: %s" % (
        "floodcaster deps checked" if (script_path.exists() and "floodcaster"
        in script_path.read_text(encoding="utf-8", errors="replace"))
        else "n/a"
    ))
    print("  7c. Import    : %s" % (
        "%d ecosystem modules verified" % len(modules_to_test)
        if modules_to_test else "no ecosystem imports"
    ))
    print("  8. Pre-install: (none)")
    print("  9. Timeout    : %ds (%.1fh)" % (timeout_s, timeout_s / 3600))

    if blockers:
        print("-" * 64)
        for b in blockers:
            print("  BLOCK: %s" % b)
    if warnings:
        print("-" * 64)
        for w in warnings:
            print("  WARN: %s" % w)
    if not blockers and not warnings:
        print("-" * 64)
        print("  ALL CLEAR")
    print("=" * 64)
    print()

    return warnings, blockers


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
    phase_id: str | None = None,
    scenario: str | None = None,
    allow_instance_override: bool = False,
    timeout_s: int = 7200,
) -> str:
    """Upload code and launch a SageMaker Processing job.

    Args:
        pip_packages: Space-separated packages to install *in addition to*
            _BASE_PACKAGES.  Pass e.g. ``"geopandas rasterio"`` for spatial
            jobs.  None means base packages only.
        image_uri: Override the default SKLEARN_CPU image. Pass PYTORCH_CPU
            for jobs that import torch, or PYTORCH_GPU for CUDA forward pass.
        env_overrides: Additional environment variables to inject into the
            container (e.g. {"EARTHDATA_TOKEN": token}).  Merged on top of
            the default environment; caller-supplied values take precedence.
        pre_install_cmd: Shell command to run before pip install (e.g. apt-get
            for system-level dependencies like libgdal-dev).
        allow_instance_override: If True, bypass the instance allowlist.
            Must be set explicitly per-launcher when GPU instances are needed.

    Raises:
        ValueError: If instance_type is not in ALLOWED_INSTANCES and
            allow_instance_override is False.
    """
    if not allow_instance_override and instance_type not in ALLOWED_INSTANCES:
        raise ValueError(
            f"Instance type '{instance_type}' not in allowlist "
            f"{sorted(ALLOWED_INSTANCES)}. Set allow_instance_override=True "
            f"or add to ALLOWED_INSTANCES in _launcher_base.py."
        )

    # Experiment contract preflight
    if phase_id:
        ok = _run_preflight(phase_id, scenario)
        if not ok and not dry_run:
            log.error("PREFLIGHT FAILED for phase=%s scenario=%s. Aborting.", phase_id, scenario)
            sys.exit(1)
        elif not ok:
            log.warning("[DRY RUN] PREFLIGHT FAILED. Fix before real launch.")

    # Resource audit (9 dimensions)
    audit_warnings, audit_blockers = _resource_audit(
        job_script=job_script,
        instance_type=instance_type,
        volume_size_gb=volume_size_gb,
        pip_packages=pip_packages,
        image_uri=image_uri,
        timeout_s=timeout_s,
    )
    if audit_blockers:
        log.error("Resource audit BLOCKED launch with %d issue(s):", len(audit_blockers))
        for b in audit_blockers:
            log.error("  BLOCK: %s", b)
        if not dry_run:
            sys.exit(1)
        else:
            log.error("[DRY RUN] Launch would be BLOCKED. Fix before real launch.")
    if audit_warnings and not dry_run:
        log.warning("Resource audit has %d warning(s). Review above.", len(audit_warnings))

    # Capture git provenance
    git_info = _get_git_info()
    log.info("Git: %s (dirty=%s)", git_info["git_hash"][:12], git_info["git_dirty"])
    if git_info["git_dirty"] == "true":
        log.warning("Working tree is dirty. Commit before launch for reproducibility.")

    code_prefix = upload_code(job_name, job_script, extra_files)

    # Write git provenance alongside code on S3
    provenance = {
        "job_name": job_name,
        "job_script": job_script,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        **git_info,
    }
    s3_prov = boto3.client("s3", **get_aws_credentials())
    s3_prov.put_object(
        Bucket=CODE_BUCKET,
        Key=f"{code_prefix}provenance.json",
        Body=json.dumps(provenance, indent=2).encode(),
        ContentType="application/json",
    )

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
        {
            "InputName": "wheels",
            "S3Input": {
                "S3Uri": f"s3://{WHEELS_BUCKET}/{WHEELS_PREFIX}/",
                "LocalPath": "/opt/ml/processing/input/wheels",
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
        "# Ecosystem wheels (yrsn, rsct, etc.) from S3 mount\n"
        "echo '--- Installing ecosystem wheels ---'\n"
        "for whl in /opt/ml/processing/input/wheels/*.whl; do\n"
        "  echo \"Installing: $whl\"\n"
        "  pip install --no-deps -q \"$whl\" || echo \"WARN: failed to install $whl\"\n"
        "done\n"
        "# Vendored wheels (swarm_auth etc.) bundled with code\n"
        "for whl in /opt/ml/processing/input/code/*.whl; do\n"
        "  [ -f \"$whl\" ] && pip install --no-deps -q \"$whl\" || true\n"
        "done\n"
        "echo '--- Verifying key packages ---'\n"
        "python -c \"import georsct; print('georsct OK:', georsct.__file__)\" || echo 'FAIL: georsct not importable'\n"
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
            "ImageUri": image_uri or DEFAULT_IMAGE,
            "ContainerEntrypoint": container_cmd,
        },
        "ProcessingInputs": processing_inputs,
        **( {"ProcessingOutputConfig": {"Outputs": processing_outputs}} if processing_outputs else {} ),
        "RoleArn": SAGEMAKER_ROLE,
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "AWS_DEFAULT_REGION": REGION,
            "S035_GIT_HASH": git_info["git_hash"],
            "S035_GIT_DIRTY": git_info["git_dirty"],
            # Route all HF Hub + Datasets cache to EBS (/tmp), not root fs (~20 GB limit)
            "HF_HOME": "/tmp/hf_cache",
            "HF_DATASETS_CACHE": "/tmp/hf_cache/datasets",
            **(env_overrides or {}),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": timeout_s},
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
