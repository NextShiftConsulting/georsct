#!/usr/bin/env python3
"""Validate Data Lock A completeness for the Houston (Harvey 2017) scenario.

LOCAL script -- checks S3 for required datasets, file counts, and sizes.
Prints a PASS/FAIL scorecard and exits 0 (all pass) or 1 (any fail).

Usage:
    python validate_data_lock_a.py
"""

import sys
from typing import Any

from swarm_auth import get_aws_credentials

import boto3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGION = "us-east-1"
BUCKET = "swarm-floodrsct-data"

HOUSTON_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "HRRR QPF": {
        "prefix": "raw/noaa_hrrr/harvey2017/",
        "min_files": 50,
        "min_file_size_bytes": 10_000_000,
    },
    "MRMS Stage IV": {
        "prefix": "raw/noaa_mrms/harvey2017/",
        "min_files": 100,
        "min_file_size_bytes": 10_000,
    },
    "NOAA Tides": {
        "prefix": "raw/noaa_tides/harvey2017/",
        "min_files": 3,
        "min_file_size_bytes": 1_000,
    },
    "Surge HWMs": {
        "prefix": "raw/surge_estimates/harvey2017/",
        "min_files": 1,
        "min_file_size_bytes": 1_000,
    },
    "DEM": {
        "prefix": "raw/dem/3dep/v1/",
        "min_files": 10,
        "min_file_size_bytes": 100_000,
    },
    "FloodSimBench": {
        "prefix": "raw/floodsimbench/6hr_max/",
        "min_files": 5,
        "min_file_size_bytes": 1_000,
    },
    "USGS STN HWMs": {
        "prefix": "raw/usgs_stn/harvey2017",
        "min_files": 1,
        "min_file_size_bytes": 1_000,
    },
    "Prithvi-EO-2.0": {
        "prefix": "model/prithvi_eo2/weights/",
        "min_files": 1,
        "min_file_size_bytes": 100_000_000,
    },
    "HURDAT2": {
        "prefix": "raw/hurdat2/",
        "min_files": 1,
        "min_file_size_bytes": 1_000,
    },
    "NLCD Impervious": {
        "prefix": "raw/nlcd/impervious_2021/",
        "min_files": 1,
        "min_file_size_bytes": 100_000_000,
    },
}

# Manifests that should exist for Data Lock A
MANIFEST_PREFIXES = [
    "manifests/noaa_mrms/",
    "manifests/noaa_tides/",
    "manifests/surge_hwm/",
    "manifests/hurdat2/",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _list_objects(s3, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns list of {Key, Size}."""
    objects: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"Key": obj["Key"], "Size": obj["Size"]})
    return objects


def _check_dataset(s3, name: str, req: dict) -> dict:
    """Check a single dataset against requirements.

    Returns:
        dict with name, status (PASS/FAIL), file_count, smallest_file,
        total_bytes, and details string.
    """
    prefix = req["prefix"]
    min_files = req["min_files"]
    min_size = req["min_file_size_bytes"]

    objects = _list_objects(s3, prefix)
    file_count = len(objects)
    total_bytes = sum(o["Size"] for o in objects)
    smallest = min((o["Size"] for o in objects), default=0)

    issues: list[str] = []

    if file_count < min_files:
        issues.append(
            f"file count {file_count} < required {min_files}"
        )

    undersized = [o for o in objects if o["Size"] < min_size]
    if undersized:
        issues.append(
            f"{len(undersized)} file(s) below min size "
            f"({min_size:,} bytes)"
        )

    status = "FAIL" if issues else "PASS"
    details = "; ".join(issues) if issues else "OK"

    return {
        "name": name,
        "prefix": prefix,
        "status": status,
        "file_count": file_count,
        "total_mb": total_bytes / 1e6,
        "smallest_bytes": smallest,
        "details": details,
    }


def _check_manifests(s3) -> list[dict]:
    """Check that expected manifest files exist."""
    results: list[dict] = []
    for prefix in MANIFEST_PREFIXES:
        objects = _list_objects(s3, prefix)
        manifest_files = [o for o in objects if o["Key"].endswith("manifest.json")]
        status = "PASS" if manifest_files else "FAIL"
        details = (
            f"{len(manifest_files)} manifest(s) found"
            if manifest_files
            else "no manifest.json found"
        )
        results.append({
            "name": f"Manifest: {prefix}",
            "prefix": prefix,
            "status": status,
            "file_count": len(manifest_files),
            "total_mb": 0.0,
            "smallest_bytes": 0,
            "details": details,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Data Lock A Validation -- Houston (Harvey 2017)")
    print(f"Bucket: s3://{BUCKET}")
    print("=" * 70)

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name=REGION, **_aws)

    # Check datasets
    results: list[dict] = []
    for name, req in HOUSTON_REQUIREMENTS.items():
        print(f"\nChecking {name} ({req['prefix']}) ...")
        r = _check_dataset(s3, name, req)
        results.append(r)
        symbol = "PASS" if r["status"] == "PASS" else "FAIL"
        print(
            f"  [{symbol}] {r['file_count']} files, "
            f"{r['total_mb']:.1f} MB -- {r['details']}"
        )

    # Check manifests
    print("\nChecking manifests ...")
    manifest_results = _check_manifests(s3)
    for r in manifest_results:
        symbol = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{symbol}] {r['name']} -- {r['details']}")
    results.extend(manifest_results)

    # Scorecard
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    print(f"\n{'=' * 70}")
    print("DATA LOCK A SCORECARD")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<22s} {'Status':<6s} {'Files':>6s} {'Size MB':>10s}  Details")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<22s} {r['status']:<6s} {r['file_count']:>6d} "
            f"{r['total_mb']:>10.1f}  {r['details']}"
        )
    print("-" * 70)
    print(f"Total: {total} | PASS: {pass_count} | FAIL: {fail_count}")

    if fail_count == 0:
        print("\nData Lock A: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"\nData Lock A: {fail_count} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
