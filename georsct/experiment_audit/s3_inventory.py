"""S3 inventory client for experiment audit artifact checks."""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .models import ArtifactRecord


@runtime_checkable
class S3ClientProtocol(Protocol):
    """Minimal S3 client interface for dependency injection."""

    def head_object(self, *, Bucket: str, Key: str) -> dict: ...

    def get_object(self, *, Bucket: str, Key: str) -> dict: ...

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **kwargs: Any) -> dict: ...


_TIMESTAMP_FIELDS = ("timestamp", "created_at", "generated_at", "run_started_at")


def _extract_internal_timestamp(data: dict) -> str | None:
    """Return the first non-None timestamp from priority-ordered fields."""
    for field in _TIMESTAMP_FIELDS:
        value = data.get(field)
        if value is not None:
            return str(value)
    return None


def _make_default_client() -> S3ClientProtocol:
    """Create an S3 client using swarm_auth credentials."""
    from swarm_auth import get_aws_credentials
    import boto3

    creds = get_aws_credentials()
    return boto3.client("s3", **creds)


class S3Inventory:
    """Wrapper around an S3 client for artifact existence and metadata checks.

    Args:
        client: An S3-compatible client. If None, one is created via swarm_auth.
        bucket: The S3 bucket name to query.
    """

    def __init__(
        self,
        client: S3ClientProtocol | None = None,
        bucket: str = "swarm-floodrsct-data",
    ) -> None:
        self._client = client if client is not None else _make_default_client()
        self._bucket = bucket

    def check_key(self, key: str) -> ArtifactRecord:
        """Check whether a key exists and return its metadata.

        Args:
            key: The S3 object key to check.

        Returns:
            ArtifactRecord with exists, size_bytes, and last_modified populated.
        """
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return ArtifactRecord(s3_key=key, exists=False)
            raise

        last_modified = resp.get("LastModified")
        if isinstance(last_modified, datetime):
            last_modified = last_modified.isoformat()
        elif last_modified is not None:
            last_modified = str(last_modified)

        return ArtifactRecord(
            s3_key=key,
            exists=True,
            size_bytes=resp.get("ContentLength"),
            last_modified=last_modified,
        )

    def download_json(self, key: str) -> ArtifactRecord:
        """Download a JSON object, parse it, and extract internal timestamp.

        Args:
            key: The S3 object key to download.

        Returns:
            ArtifactRecord with content and internal_timestamp populated.
        """
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return ArtifactRecord(s3_key=key, exists=False)
            raise

        body = resp["Body"]
        if hasattr(body, "read"):
            raw = body.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
        else:
            raw = body

        data = json.loads(raw)

        last_modified = resp.get("LastModified")
        if isinstance(last_modified, datetime):
            last_modified = last_modified.isoformat()
        elif last_modified is not None:
            last_modified = str(last_modified)

        return ArtifactRecord(
            s3_key=key,
            exists=True,
            size_bytes=resp.get("ContentLength"),
            last_modified=last_modified,
            internal_timestamp=_extract_internal_timestamp(data),
            content=data,
        )

    def list_prefix(self, prefix: str) -> list[str]:
        """List all keys under a prefix, handling pagination.

        Args:
            prefix: The S3 key prefix to list.

        Returns:
            A list of matching S3 keys.
        """
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._bucket,
                "Prefix": prefix,
            }
            if continuation_token is not None:
                kwargs["ContinuationToken"] = continuation_token

            resp = self._client.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])

            if not resp.get("IsTruncated", False):
                break
            continuation_token = resp.get("NextContinuationToken")

        return keys


def _is_not_found(exc: Exception) -> bool:
    """Check if an exception represents a 404 / NoSuchKey error."""
    error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return error_code in ("404", "NoSuchKey")
