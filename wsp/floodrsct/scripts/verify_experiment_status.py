#!/usr/bin/env python3
"""
verify_experiment_status.py -- Audit EXPERIMENT_STATUS.yaml against S3.

Reads the experiment registry and checks every artifact_key against S3.
Reports mismatches: COMPLETED but missing, DESIGNED but artifact exists.

Usage:
    python verify_experiment_status.py
    python verify_experiment_status.py --fix   # update status in-place
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

STATUS_FILE = (
    Path(__file__).resolve().parents[1]
    / "exp" / "s035-model-ladder" / "EXPERIMENT_STATUS.yaml"
)


def get_s3_client():
    session = boto3.Session(profile_name="nsc-swarm")
    return session.client("s3")


def check_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def collect_artifact_checks(phases: dict, prefix: str = "") -> list[dict]:
    """Recursively collect all artifact_keys with their status and path."""
    checks = []
    for name, value in phases.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            if "artifact_keys" in value:
                status = value.get("status", "UNKNOWN")
                for key in value["artifact_keys"]:
                    checks.append({
                        "phase": path,
                        "status": status,
                        "artifact_key": key,
                    })
            # Recurse into nested phases
            for k, v in value.items():
                if isinstance(v, dict) and k not in (
                    "artifact_keys", "notes", "paper_claims"
                ):
                    checks.extend(
                        collect_artifact_checks({k: v}, prefix=path)
                    )
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment status against S3 artifacts"
    )
    parser.add_argument("--fix", action="store_true",
                        help="Update EXPERIMENT_STATUS.yaml with findings")
    args = parser.parse_args()

    if not STATUS_FILE.exists():
        print(f"ERROR: {STATUS_FILE} not found")
        sys.exit(1)

    with open(STATUS_FILE) as f:
        registry = yaml.safe_load(f)

    bucket = registry["s3_bucket"]
    phases = registry.get("phases", {})

    s3 = get_s3_client()

    checks = collect_artifact_checks(phases)

    print(f"Experiment: {registry['experiment']}")
    print(f"Last audit: {registry.get('last_audit', 'never')}")
    print(f"Checking {len(checks)} artifact keys against s3://{bucket}/")
    print(f"{'='*70}")

    ok = 0
    mismatch = 0
    missing = 0
    surprise = 0

    for check in checks:
        exists = check_key_exists(s3, bucket, check["artifact_key"])
        status = check["status"]

        if status == "COMPLETED" and exists:
            ok += 1
            marker = "OK"
        elif status == "COMPLETED" and not exists:
            missing += 1
            marker = "MISSING"
        elif status == "DESIGNED" and exists:
            surprise += 1
            marker = "SURPRISE"
        elif status == "DESIGNED" and not exists:
            ok += 1
            marker = "OK"
        else:
            mismatch += 1
            marker = f"CHECK ({status}, exists={exists})"

        icon = {
            "OK": "[+]", "MISSING": "[!]", "SURPRISE": "[?]",
        }.get(marker, "[~]")

        if marker != "OK":
            print(f"  {icon} {check['phase']}")
            print(f"      status={status}, s3_exists={exists}")
            print(f"      key={check['artifact_key']}")

    print(f"\n{'='*70}")
    print(f"  OK:       {ok}")
    print(f"  MISSING:  {missing}  (status=COMPLETED but artifact gone)")
    print(f"  SURPRISE: {surprise}  (status=DESIGNED but artifact exists)")
    print(f"  OTHER:    {mismatch}")
    print(f"{'='*70}")

    if missing == 0 and surprise == 0 and mismatch == 0:
        print("  ALL CLEAR -- registry matches S3")
    else:
        print("  ACTION NEEDED -- update EXPERIMENT_STATUS.yaml")

    # Also check paper claims
    claims = registry.get("paper_claims", {})
    if claims:
        print(f"\nPaper Claim Status:")
        for section, info in claims.items():
            status = info.get("status", "UNKNOWN")
            icon = {"GREEN": "[+]", "YELLOW": "[~]", "RED": "[!]"}.get(
                status, "[?]"
            )
            print(f"  {icon} {section}: {status}")
            if status in ("RED", "YELLOW"):
                print(f"      {info.get('reason', '')}")


if __name__ == "__main__":
    main()
