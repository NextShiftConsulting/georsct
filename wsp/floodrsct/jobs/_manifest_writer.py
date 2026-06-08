"""
_manifest_writer.py -- Shared utility: write a dataset manifest.json to S3.

Every raw fetch job should call write_manifest() before exiting.
Manifests are the provenance record for each raw dataset version.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import boto3

log = logging.getLogger(__name__)

MANIFEST_BUCKET = "swarm-floodrsct-data"
MANIFEST_PREFIX = "manifests"


def write_manifest(
    s3,
    dataset: str,
    version: str,
    source_url: str,
    s3_key: str,
    crs: str = "EPSG:4326",
    record_count: Optional[int] = None,
    etag: Optional[str] = None,
    sha256: Optional[str] = None,
    license_: str = "Public Domain / U.S. Government",
    notes: str = "",
) -> str:
    """Write a manifest.json to s3://swarm-floodrsct-data/manifests/{dataset}/{version}/."""
    manifest = {
        "dataset": dataset,
        "version": version,
        "source_url": source_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "s3_bucket": MANIFEST_BUCKET,
        "s3_key": s3_key,
        "etag": etag,
        "sha256": sha256,
        "crs": crs,
        "record_count": record_count,
        "license": license_,
        "notes": notes,
    }

    manifest_key = f"{MANIFEST_PREFIX}/{dataset}/{version}/manifest.json"
    payload = json.dumps(manifest, indent=2).encode()
    s3.put_object(
        Bucket=MANIFEST_BUCKET,
        Key=manifest_key,
        Body=payload,
        ContentType="application/json",
    )
    log.info("Manifest written to s3://%s/%s", MANIFEST_BUCKET, manifest_key)
    return manifest_key
