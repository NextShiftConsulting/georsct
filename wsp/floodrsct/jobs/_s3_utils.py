"""
_s3_utils.py -- Reusable S3 helpers for DataFrame I/O via swarm_auth.

Builds on _s3_stream.py (thread-local client, streaming downloads) and adds
DataFrame-level convenience: read/write parquet, check existence.

All S3 access uses swarm_auth.get_aws_credentials(). No bare boto3.

Usage:
    from _s3_utils import get_s3, upload_parquet, download_parquet

    s3 = get_s3()
    upload_parquet(s3, df, "swarm-floodrsct-data", "processed/s040a/file.parquet")
    df = download_parquet(s3, "swarm-floodrsct-data", "raw/openfema/file.parquet")
"""

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd

# Re-export thread-local S3 client from _s3_stream
from _s3_stream import get_s3, s3_key_exists  # noqa: F401

log = logging.getLogger(__name__)


def upload_parquet(
    s3,
    df: pd.DataFrame,
    bucket: str,
    key: str,
    *,
    tmp_dir: str = "/tmp",
) -> None:
    """Write DataFrame to parquet and upload to S3.

    Args:
        s3: boto3 S3 client (from get_s3()).
        df: DataFrame to upload.
        bucket: S3 bucket name.
        key: S3 object key (e.g. "processed/s040a/file.parquet").
        tmp_dir: Local temp directory for staging.
    """
    local = str(Path(tmp_dir) / Path(key).name)
    df.to_parquet(local, index=False)
    s3.upload_file(local, bucket, key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), bucket, key)


def download_parquet(
    s3,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    """Download a parquet file from S3 and return as DataFrame.

    Returns empty DataFrame if key does not exist.

    Args:
        s3: boto3 S3 client (from get_s3()).
        bucket: S3 bucket name.
        key: S3 object key.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        log.info("Downloaded %d rows from s3://%s/%s", len(df), bucket, key)
        return df
    except s3.exceptions.NoSuchKey:
        log.warning("Key not found: s3://%s/%s", bucket, key)
        return pd.DataFrame()
