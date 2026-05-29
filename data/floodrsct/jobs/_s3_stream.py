"""
_s3_stream.py -- Checkpointed streaming download -> S3 helper for s035 jobs.

Pattern:
  1. s3_key_exists() -- skip if already uploaded (resume support)
  2. stream_to_tmp() -- HTTP stream -> disk, never loads file into RAM
  3. upload_file()   -- boto3 auto-multipart for files > 8 MB
  4. cleanup         -- delete tmp file

Thread safety:
  - get_s3() returns a thread-local boto3 client
  - Each tmp file uses a uuid name -- no collision across concurrent threads
  - Pass s3=None to use the thread-local client

Container vs local:
  - All environments use swarm_auth.get_aws_credentials() for credential discovery
  - swarm_auth handles IAM roles (SageMaker) and local profiles transparently
"""

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import boto3
import requests
from swarm_auth import get_aws_credentials

_tls = threading.local()


def get_s3():
    """Return a thread-local S3 client via swarm_auth credentials."""
    if not hasattr(_tls, "s3"):
        _aws = get_aws_credentials()
        _aws.pop("region_name", None)
        _tls.s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    return _tls.s3


log = logging.getLogger(__name__)


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    """Return True if the key already exists in S3 (checkpoint skip)."""
    client = s3 or get_s3()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def stream_to_tmp(url: str, tmp_path: str, retries: int = 3, timeout: int = 300) -> bool:
    """Stream an HTTP URL to a local temp file. Returns True on success."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                    fh.write(chunk)
            return True
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return False


def stream_download_to_s3(
    s3,
    url: str,
    bucket: str,
    key: str,
    tmp_dir: str = "/tmp",
    retries: int = 3,
    timeout: int = 300,
    min_size_bytes: int = 0,
    extra_args: Optional[dict] = None,
) -> bool:
    """Download url and upload to s3://bucket/key. Skip if already exists.

    Args:
        min_size_bytes: If > 0, reject downloads smaller than this (catches
            error pages returned with HTTP 200).

    Returns True if key is now present in S3.
    """
    client = s3 or get_s3()

    if s3_key_exists(client, bucket, key):
        log.debug("Checkpoint hit -- already in S3: s3://%s/%s", bucket, key)
        return True

    tmp_name = f"{uuid.uuid4().hex}_{Path(key).name}"
    tmp_path = str(Path(tmp_dir) / tmp_name)

    try:
        ok = stream_to_tmp(url, tmp_path, retries=retries, timeout=timeout)
        if not ok:
            return False

        file_size = Path(tmp_path).stat().st_size
        if min_size_bytes > 0 and file_size < min_size_bytes:
            log.warning(
                "Downloaded file too small (%d bytes < %d min): %s",
                file_size, min_size_bytes, url,
            )
            return False

        size_mb = file_size / 1e6
        log.info("Uploading s3://%s/%s (%.1f MB)", bucket, key, size_mb)

        upload_kwargs = {}
        if extra_args:
            upload_kwargs["ExtraArgs"] = extra_args

        client.upload_file(tmp_path, bucket, key, **upload_kwargs)
        return True

    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
