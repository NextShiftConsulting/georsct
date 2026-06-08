"""
preflight.py -- S3 artifact pre-flight checks for georsct-rerun SageMaker launchers.

Copied from yrsn-experiments/exp/series_018/shared/preflight.py with
only the import path changed (this file is self-contained in georsct-rerun).
"""

import io
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


@dataclass
class S3Artifact:
    """A single S3 object to verify."""
    bucket: str
    key: str
    description: str = ""
    required_keys: Optional[List[str]] = None
    required_columns: Optional[List[str]] = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass
class ArtifactGroup:
    """Named group of S3 artifacts (for readable output)."""
    name: str
    artifacts: List[S3Artifact]


def _check_artifact(s3_client, artifact: S3Artifact) -> tuple[bool, str]:
    """Return (exists, message) for one S3 artifact."""
    try:
        s3_client.head_object(Bucket=artifact.bucket, Key=artifact.key)
        return True, f"  OK  {artifact.uri}"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False, f"  MISSING  {artifact.uri}"
        return False, f"  ERROR ({code})  {artifact.uri} -- {e}"


def _probe_schema(s3_client, artifact: S3Artifact) -> tuple[bool, List[str]]:
    """Probe internal schema of an artifact."""
    messages = []
    if not artifact.required_keys and not artifact.required_columns:
        return True, []

    try:
        with tempfile.NamedTemporaryFile(suffix=Path(artifact.key).suffix, delete=False) as tmp:
            s3_client.download_fileobj(artifact.bucket, artifact.key, tmp)
            tmp_path = tmp.name
    except Exception as e:
        return False, [f"    PROBE ERROR: could not download for schema check: {e}"]

    ok = True

    if artifact.required_keys and artifact.key.endswith(".npz"):
        try:
            import numpy as np
            art = np.load(tmp_path, allow_pickle=True)
            available = set(art.files)
            missing = [k for k in artifact.required_keys if k not in available]
            if missing:
                ok = False
                messages.append(
                    f"    SCHEMA FAIL: .npz missing keys {missing} "
                    f"(has: {sorted(available)})"
                )
            else:
                messages.append(
                    f"    SCHEMA OK: .npz keys {artifact.required_keys} present"
                )
        except Exception as e:
            ok = False
            messages.append(f"    PROBE ERROR: could not read .npz: {e}")

    if artifact.required_columns and artifact.key.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(tmp_path)
            available = set(schema.names)
            missing = [c for c in artifact.required_columns if c not in available]
            if missing:
                ok = False
                messages.append(
                    f"    SCHEMA FAIL: .parquet missing columns {missing}"
                )
            else:
                messages.append(
                    f"    SCHEMA OK: .parquet columns {artifact.required_columns} present"
                )
        except Exception as e:
            ok = False
            messages.append(f"    PROBE ERROR: could not read .parquet schema: {e}")

    try:
        Path(tmp_path).unlink()
    except OSError:
        pass

    return ok, messages


AWS_PROFILE = "nsc-swarm"


def preflight_check(
    groups: List[ArtifactGroup],
    region: str = "us-east-1",
    dry_run: bool = False,
    profile_name: str = AWS_PROFILE,
) -> bool:
    """Verify all artifact groups exist on S3."""
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")

    print("=" * 60)
    print("PRE-FLIGHT S3 ARTIFACT CHECK")
    print("=" * 60)

    all_ok = True
    missing = []
    schema_failures = []

    for group in groups:
        print(f"\n[{group.name}]")
        for artifact in group.artifacts:
            ok, msg = _check_artifact(s3, artifact)
            print(msg)
            if not ok:
                all_ok = False
                missing.append(artifact)
            elif artifact.required_keys or artifact.required_columns:
                probe_ok, probe_msgs = _probe_schema(s3, artifact)
                for m in probe_msgs:
                    print(m)
                if not probe_ok:
                    all_ok = False
                    schema_failures.append(artifact)

    print("\n" + "=" * 60)
    if all_ok:
        print("PRE-FLIGHT PASSED -- all artifacts present and schemas valid")
    else:
        if missing:
            print(f"PRE-FLIGHT FAILED -- {len(missing)} artifact(s) missing:")
            for a in missing:
                print(f"  {a.uri}")
        if schema_failures:
            print(f"PRE-FLIGHT FAILED -- {len(schema_failures)} artifact(s) have schema errors:")
            for a in schema_failures:
                print(f"  {a.uri}")
        print("\nFix missing/invalid artifacts before launching. Job NOT started.")
    print("=" * 60)

    return all_ok


# ---------------------------------------------------------------------------
# Canonical artifact sets
# ---------------------------------------------------------------------------

BUCKET = "swarm-yrsn-datasets"
WHEEL_PREFIX = "rsct_code/wheels/20260506-162534"


def wheels_group(require_controlplane: bool = True) -> ArtifactGroup:
    """Standard wheel artifacts."""
    artifacts = [
        S3Artifact(BUCKET, f"{WHEEL_PREFIX}/yrsn-1.0.0-py3-none-any.whl"),
    ]
    if require_controlplane:
        artifacts.append(
            S3Artifact(BUCKET, f"{WHEEL_PREFIX}/yrsn_controlplane-0.1.0-py3-none-any.whl")
        )
    return ArtifactGroup("wheels", artifacts)
