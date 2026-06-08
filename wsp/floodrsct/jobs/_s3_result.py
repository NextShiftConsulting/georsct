"""
_s3_result.py -- Archive-safe S3 result upload for s035.

Every canonical result file (r0_houston.json, diagnostics_r0.json, etc.)
is immutable evidence.  Before overwriting, the previous version is archived
with a timestamp + git hash in the key so no result is ever silently lost.

Usage:
    from _s3_result import upload_json_result
    upload_json_result(s3, bucket, key, payload_dict)
"""

import json
import logging
import os
import re
from datetime import date, datetime, timezone

import numpy as np
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

def _sanitize_nan(obj):
    """Replace float NaN/Inf with None so JSON stays RFC-8259 compliant.

    Python's json.dumps emits bare NaN/Infinity which is invalid JSON.
    Consumers already handle None (null) — bare NaN causes silent data
    corruption downstream (e.g., pooled Wilcoxon NaN poisoning).
    """
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


class ResultJSONEncoder(json.JSONEncoder):
    """Strict encoder: numpy scalars/arrays and datetime/date only.

    Unknown types raise TypeError — prevents silent str() coercion of
    arbitrary objects into unreadable payload junk.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return _sanitize_nan(obj.tolist())
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


# S3 keys must be ASCII-safe.  Colons from ISO timestamps are replaced.
_UNSAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._\-/]")


def _s3_safe_ts(ts: str) -> str:
    """Convert an ISO timestamp to an S3-key-safe string.

    >>> _s3_safe_ts("2026-06-02T14:30:00+00:00")
    '20260602T143000Z'
    """
    # Strip sub-second and timezone suffix, keep digits + T
    clean = ts.replace("-", "").replace(":", "").replace("+00:00", "Z").replace("+0000", "Z")
    # Remove any remaining unsafe chars
    clean = _UNSAFE_KEY_RE.sub("", clean)
    # Truncate to reasonable length
    return clean[:20]


def upload_json_result(
    s3,
    bucket: str,
    key: str,
    payload: dict,
    *,
    git_hash: str | None = None,
) -> None:
    """Upload a JSON result to S3, archiving any pre-existing version.

    Archive key format:
        {parent}/archive/{stem}_{timestamp}_{git_hash:.8}.json

    Args:
        s3: boto3 S3 client.
        bucket: S3 bucket name.
        key: Canonical result key (e.g. "results/s035/r0_houston.json").
        payload: Dict to serialize as JSON.
        git_hash: Git commit hash for provenance.  Falls back to
            S035_GIT_HASH env var, then "unknown".

    Raises:
        ClientError: On permission errors (AccessDenied, etc.) during
            archive read or write.  NoSuchKey on read is silently allowed
            (first run).
        TypeError / ValueError: If payload is not JSON-serializable.
    """
    if git_hash is None:
        git_hash = os.environ.get("S035_GIT_HASH", "unknown")

    # Sanitize NaN/Inf -> None before serialization (RFC-8259 compliance)
    payload = _sanitize_nan(payload)

    # Serialize first — fail fast on bad payload before touching S3
    body = json.dumps(payload, indent=2, cls=ResultJSONEncoder).encode()

    # Archive existing result (if any)
    _archive_existing(s3, bucket, key, git_hash)

    # Write canonical result
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    log.info("Uploaded s3://%s/%s", bucket, key)


def _archive_existing(s3, bucket: str, key: str, git_hash: str) -> None:
    """Copy the existing object at *key* to an archive key.

    Tolerates NoSuchKey (first write).  Re-raises any other ClientError
    (AccessDenied, bucket-not-found, etc.) so callers see real failures.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        old_body = resp["Body"].read()
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            return  # First write — nothing to archive
        raise  # AccessDenied, etc. — caller must see this

    # Extract old timestamp for the archive key
    try:
        old_meta = json.loads(old_body)
        old_ts = old_meta.get("timestamp", "unknown")
    except (json.JSONDecodeError, UnicodeDecodeError):
        old_ts = "unparseable"

    ts_tag = _s3_safe_ts(old_ts)
    hash_tag = git_hash[:8] if git_hash else "nohash"

    # Build archive key:  results/s035/archive/r0_houston_20260602T143000Z_abcd1234.json
    parent = key.rsplit("/", 1)[0] if "/" in key else ""
    stem = key.rsplit("/", 1)[-1]  # e.g. "r0_houston.json"
    name_no_ext = stem.removesuffix(".json")
    archive_key = f"{parent}/archive/{name_no_ext}_{ts_tag}_{hash_tag}.json"

    s3.put_object(
        Bucket=bucket,
        Key=archive_key,
        Body=old_body,
        ContentType="application/json",
    )
    log.warning("Archived previous result -> s3://%s/%s", bucket, archive_key)
