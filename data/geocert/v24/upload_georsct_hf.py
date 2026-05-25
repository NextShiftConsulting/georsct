#!/usr/bin/env python3
"""
upload_georsct_hf.py -- Rebuild enriched features parquet and push to HuggingFace.

Steps:
  1. (Optional) Wait for SageMaker jobs to complete
  2. Run build_zcta_features_labels.py to pull NOAA + NFIP enrichments from S3
     and produce updated zcta_features_labels.parquet
  3. Upload updated parquet to rudymartin/georsct as georsct_table.parquet
  4. Regenerate and upload build_manifest.json with new column list + timestamp

Usage:
    python upload_georsct_hf.py --dry-run          # local build only, no upload
    python upload_georsct_hf.py                     # build + upload to HuggingFace
    python upload_georsct_hf.py --skip-rebuild      # upload existing local parquet
    python upload_georsct_hf.py --wait-for-jobs NOAA_JOB NFIP_JOB  # poll then run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from huggingface_hub import HfApi

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"
AWS_PROFILE = "nsc-swarm"
REGION = "us-east-1"
HF_REPO_ID = "rudymartin/georsct"

LOCAL_PARQUET = Path("/tmp/georsct_table_v24.parquet")
SCRIPT_DIR = Path(__file__).parent


def wait_for_jobs(job_names: list[str], poll_interval: int = 30):
    """Poll SageMaker job statuses until all complete or one fails."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    sm = session.client("sagemaker")

    pending = set(job_names)
    print(f"Waiting for {len(pending)} SageMaker jobs...")

    while pending:
        done = set()
        for job in list(pending):
            resp = sm.describe_processing_job(ProcessingJobName=job)
            status = resp["ProcessingJobStatus"]
            print(f"  {job}: {status}")
            if status == "Completed":
                done.add(job)
            elif status in ("Failed", "Stopped"):
                print(f"  ERROR: Job {job} ended with status {status}")
                failure = resp.get("FailureReason", "unknown")
                print(f"  Reason: {failure}")
                sys.exit(1)
        pending -= done
        if pending:
            print(f"  {len(pending)} still running — sleeping {poll_interval}s...")
            time.sleep(poll_interval)

    print("All jobs completed.")


def rebuild_features_parquet(output_path: Path, dry_run: bool = False):
    """Run build_zcta_features_labels.py to pull S3 enrichments and merge."""
    script = SCRIPT_DIR / "build_zcta_features_labels.py"
    cmd = [
        sys.executable, str(script),
        "--output", str(output_path),
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n=== REBUILDING FEATURES PARQUET ===")
    print(f"  Script: {script}")
    print(f"  Output: {output_path}")
    result = subprocess.run(cmd, check=True)
    print(f"  Exit code: {result.returncode}")


def summarize_parquet(path: Path) -> dict:
    """Return summary stats for the parquet."""
    df = pd.read_parquet(path)
    cols = list(df.columns)
    enrichment_groups = {
        "acs": [c for c in cols if c.startswith("acs_")],
        "svi": [c for c in cols if c.startswith("svi_")],
        "flood": [c for c in cols if c.startswith("flood_")],
        "hifld": [c for c in cols if c.startswith("hifld_")],
        "drive": [c for c in cols if c.startswith("drive_")],
        "nfip": [c for c in cols if c.startswith("nfip_")],
        "noaa": [c for c in cols if c.startswith("flood_event") or c.startswith("flood_death")
                 or c.startswith("flood_injury") or c.startswith("flood_property")
                 or c.startswith("flood_crop") or c.startswith("flood_events_per")],
        "target": [c for c in cols if c.startswith("target_")],
    }
    return {
        "n_rows": len(df),
        "n_cols": len(cols),
        "columns": cols,
        "enrichment_counts": {k: len(v) for k, v in enrichment_groups.items()},
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        "nfip_present": len(enrichment_groups["nfip"]) > 0,
        "noaa_present": len(enrichment_groups["noaa"]) > 0,
    }


DTYPE_MAP = {
    "object": "sc:Text",
    "int64": "sc:Integer",
    "int32": "sc:Integer",
    "float64": "sc:Float",
    "float32": "sc:Float",
    "bool": "sc:Boolean",
}

FIELD_DESCRIPTIONS = {
    # NFIP
    "nfip_claim_count": "Total paid NFIP flood insurance claims (1978-present)",
    "nfip_total_building_loss": "Total building payout USD",
    "nfip_total_contents_loss": "Total contents payout USD",
    "nfip_total_loss": "Total combined payout USD",
    "nfip_mean_loss_per_claim": "Average payout per paid claim USD",
    "nfip_has_claims": "Any historical paid NFIP claim in this ZCTA",
    # NOAA
    "flood_event_count": "Total NOAA flood events 1996-2024",
    "flood_event_count_5y": "Flood events 2019-2024 (recent 5-year window)",
    "flood_deaths": "Flood-related deaths (direct + indirect) 1996-2024",
    "flood_injuries": "Flood-related injuries 1996-2024",
    "flood_property_damage_k": "Flood property damage in $1000s 1996-2024",
    "flood_crop_damage_k": "Flood crop damage in $1000s 1996-2024",
    "flood_events_per_year": "Annualized flood event rate 1996-2024",
}


def update_croissant(existing: dict, df_cols: list[str], df_dtypes: dict,
                     timestamp: str, version: str) -> dict:
    """Add new columns to the georsct-main recordSet and bump version/date."""
    import copy
    updated = copy.deepcopy(existing)
    updated["dateModified"] = timestamp[:10]
    updated["version"] = version

    main_rs = next(r for r in updated["recordSet"] if r["name"] == "georsct-main")
    existing_field_names = {f["name"] for f in main_rs["field"]}

    new_fields = []
    for col in df_cols:
        if col in existing_field_names:
            continue
        dtype_str = str(df_dtypes.get(col, "object"))
        sc_type = DTYPE_MAP.get(dtype_str, "sc:Text")
        field = {
            "@type": "cr:Field",
            "@id": f"georsct-main/{col}",
            "name": col,
            "dataType": sc_type,
        }
        desc = FIELD_DESCRIPTIONS.get(col)
        if desc:
            field["description"] = desc
        new_fields.append(field)

    main_rs["field"].extend(new_fields)
    return updated, len(new_fields)


def build_manifest(summary: dict, timestamp: str) -> dict:
    return {
        "version": "v24.001",
        "timestamp": timestamp,
        "n_rows": summary["n_rows"],
        "n_columns": summary["n_cols"],
        "enrichment_layers": {
            "acs_features": summary["enrichment_counts"]["acs"],
            "cdc_svi": summary["enrichment_counts"]["svi"],
            "fema_flood_zones": summary["enrichment_counts"]["flood"],
            "hifld_facilities": summary["enrichment_counts"]["hifld"],
            "drive_times": summary["enrichment_counts"]["drive"],
            "fema_nfip_claims": summary["enrichment_counts"]["nfip"],
            "noaa_storm_events": summary["enrichment_counts"]["noaa"],
        },
        "columns": summary["columns"],
        "file_size_mb": summary["size_mb"],
        "description": (
            "GeoRSCT ZCTA-level features and labels. "
            "v24.001 adds FEMA NFIP flood insurance claims (1978-present) "
            "and NOAA Storm Events flood history (1996-2024) as independent "
            "modal sources for flood certificate experiments."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild and upload GeoRSCT to HuggingFace")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, don't upload to HuggingFace or S3")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Skip rebuild — upload existing parquet at --local-parquet")
    parser.add_argument("--local-parquet", type=str, default=str(LOCAL_PARQUET),
                        help="Local path for the rebuilt parquet")
    parser.add_argument("--wait-for-jobs", nargs="+", metavar="JOB_NAME",
                        help="SageMaker job names to wait for before rebuilding")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between job status polls (default: 30)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_path = Path(args.local_parquet)

    # --- 1. Wait for SageMaker jobs ---
    if args.wait_for_jobs:
        wait_for_jobs(args.wait_for_jobs, poll_interval=args.poll_interval)

    # --- 2. Rebuild features parquet ---
    if not args.skip_rebuild:
        rebuild_features_parquet(output_path, dry_run=args.dry_run)
    else:
        if not output_path.exists():
            print(f"ERROR: --skip-rebuild specified but {output_path} not found")
            sys.exit(1)
        print(f"Skipping rebuild — using existing {output_path}")

    if not output_path.exists():
        print(f"ERROR: Expected parquet at {output_path} — rebuild may have failed")
        sys.exit(1)

    # --- 3. Summarize ---
    print(f"\n=== PARQUET SUMMARY ===")
    summary = summarize_parquet(output_path)
    print(f"  Rows:    {summary['n_rows']:,}")
    print(f"  Columns: {summary['n_cols']}")
    print(f"  Size:    {summary['size_mb']} MB")
    print(f"  NFIP present:  {summary['nfip_present']} ({summary['enrichment_counts']['nfip']} cols)")
    print(f"  NOAA present:  {summary['noaa_present']} ({summary['enrichment_counts']['noaa']} cols)")

    if not summary["nfip_present"]:
        print("  WARNING: NFIP columns not found — NFIP job may not have completed yet")
    if not summary["noaa_present"]:
        print("  WARNING: NOAA columns not found — NOAA job may not have completed yet")

    manifest = build_manifest(summary, timestamp)

    if args.dry_run:
        print("\n[DRY RUN] Skipping HuggingFace upload.")
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[DRY RUN] Manifest written to {manifest_path}")
        return

    # --- 4. Upload to HuggingFace ---
    print(f"\n=== UPLOADING TO HUGGINGFACE ===")
    api = HfApi()

    # Fetch and update Croissant
    print("  Updating croissant.json...")
    try:
        croissant_path = api.hf_hub_download(HF_REPO_ID, "croissant.json",
                                              repo_type="dataset")
        with open(croissant_path) as f:
            existing_croissant = json.load(f)
        df_tmp = pd.read_parquet(output_path)
        df_dtypes = {c: str(df_tmp[c].dtype) for c in df_tmp.columns}
        updated_croissant, n_new = update_croissant(
            existing_croissant, list(df_tmp.columns), df_dtypes,
            timestamp, "v24.001"
        )
        print(f"  Croissant: {n_new} new fields added")
        croissant_bytes = json.dumps(updated_croissant, indent=2).encode()
        api.upload_file(
            path_or_fileobj=croissant_bytes,
            path_in_repo="croissant.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="chore(croissant): add NFIP + NOAA fields, bump to v24.001",
        )
        print(f"  -> {HF_REPO_ID}/croissant.json")
    except Exception as exc:
        print(f"  WARNING: Croissant update failed: {exc}")

    # Main table parquet
    print(f"  Uploading georsct_table.parquet ({summary['size_mb']} MB)...")
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="georsct_table.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="feat(v24.001): add NFIP + NOAA flood enrichment layers",
    )
    print(f"  -> {HF_REPO_ID}/georsct_table.parquet")

    # Updated build manifest
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    api.upload_file(
        path_or_fileobj=manifest_bytes,
        path_in_repo="build_manifest.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="chore(manifest): update v24.001 column list and enrichment layers",
    )
    print(f"  -> {HF_REPO_ID}/build_manifest.json")

    print(f"\nDone. Dataset: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
