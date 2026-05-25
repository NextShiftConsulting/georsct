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
from swarm_auth import get_aws_credentials, get_credential

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"
REGION = "us-east-1"
HF_REPO_ID = "rudymartin/georsct"

LOCAL_PARQUET = Path("/tmp/georsct_table_v24.parquet")
SCRIPT_DIR = Path(__file__).parent


def wait_for_jobs(job_names: list[str], poll_interval: int = 30):
    """Poll SageMaker job statuses until all complete or one fails."""
    _aws = get_aws_credentials()
    sm = boto3.client("sagemaker", region_name=REGION, **_aws)

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
        "twi": [c for c in cols if c.startswith("twi_") or c.startswith("slope_")],
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
    # Metadata
    "population": "Total population (ACS 5-year estimate)",
    # Splits
    "split_extrap": "Extrapolation split assignment (train/val/test)",
    # Flood zones
    "flood_sfha": "Fraction of ZCTA land area in FEMA Special Flood Hazard Area (Zone A/AE/AH/AO/V/VE)",
    # Drive times
    "drive_min_to_county_centroid": "Estimated drive time in minutes to county population centroid",
    # TWI / watershed (StreamCat via USGS ScienceBase, area-weighted from NHDPlus COMIDs)
    "twi_twi": "Topographic Wetness Index — catchment-level mean (CAT_TWI from StreamCat)",
    "twi_acc_twi": "Topographic Wetness Index — accumulated upstream mean (ACC_TWI from StreamCat)",
    "twi_tot_twi": "Topographic Wetness Index — total watershed mean (TOT_TWI from StreamCat)",
    "slope_basin_slope": "Mean catchment basin slope in percent (CAT_BASIN_SLOPE from StreamCat)",
    "slope_stream_slope": "Mean stream slope in percent (CAT_STREAM_SLOPE from StreamCat)",
    "slope_mean_pct": "Alias for slope_basin_slope — mean catchment slope percent",
    # NFIP
    "nfip_claim_count": "Total paid NFIP flood insurance claims (1978-present)",
    "nfip_total_building_loss": "Total building loss payout USD (1978-present)",
    "nfip_total_contents_loss": "Total contents loss payout USD (1978-present)",
    "nfip_total_loss": "Total combined building + contents payout USD (1978-present)",
    "nfip_mean_loss_per_claim": "Average payout per paid claim USD",
    "nfip_has_claims": "Any historical paid NFIP claim in this ZCTA (1978-present)",
    # NOAA storm events
    "flood_event_count": "Total NOAA flood events 1996-2024",
    "flood_event_count_5y": "Flood events 2019-2024 (recent 5-year window)",
    "flood_deaths": "Flood-related deaths (direct + indirect) 1996-2024",
    "flood_injuries": "Flood-related injuries 1996-2024",
    "flood_property_damage_k": "Flood property damage in $1000s 1996-2024",
    "flood_crop_damage_k": "Flood crop damage in $1000s 1996-2024",
    "flood_events_per_year": "Annualized flood event rate 1996-2024",
}


def reconcile_croissant(existing: dict, df_cols: list[str], df_dtypes: dict,
                        timestamp: str, version: str) -> tuple[dict, dict]:
    """
    Reconcile georsct-main recordSet to exactly match df_cols:
      - Remove fields not in df_cols (ghost fields from prior dataset versions)
      - Add fields in df_cols not already in Croissant
      - Backfill descriptions on existing fields that lack them
      - Bump version and dateModified
      - Update dataset description to reflect current column counts

    Returns (updated_croissant, stats_dict).
    """
    import copy
    updated = copy.deepcopy(existing)
    updated["dateModified"] = timestamp[:10]
    updated["version"] = version

    col_set = set(df_cols)
    main_rs = next(r for r in updated["recordSet"] if r["name"] == "georsct-main")

    # 1. Prune ghost fields (in Croissant but not in parquet)
    before = len(main_rs["field"])
    main_rs["field"] = [f for f in main_rs["field"] if f["name"] in col_set]
    n_pruned = before - len(main_rs["field"])

    # 2. Backfill missing descriptions on surviving fields
    n_desc_added = 0
    for f in main_rs["field"]:
        if not f.get("description") and f["name"] in FIELD_DESCRIPTIONS:
            f["description"] = FIELD_DESCRIPTIONS[f["name"]]
            n_desc_added += 1

    # 3. Add fields in parquet not yet in Croissant
    existing_names = {f["name"] for f in main_rs["field"]}
    n_added = 0
    for col in df_cols:
        if col in existing_names:
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
        main_rs["field"].append(field)
        n_added += 1

    # 4. Update dataset-level description
    n_acs    = sum(1 for c in df_cols if c.startswith("acs_"))
    n_target = sum(1 for c in df_cols if c.startswith("target_"))
    n_enrich = len(df_cols) - n_acs - n_target
    updated["description"] = (
        f"A geospatial regression benchmark for evaluating representation-solver compatibility. "
        f"Contains 31,789 U.S. ZIP Code Tabulation Areas (ZCTAs) across the contiguous United States "
        f"(lower 48 + DC) with {n_acs} ACS socioeconomic features, {n_enrich} enrichment features "
        f"(SVI, HIFLD, FEMA flood zones, drive times, NOAA storm events, NFIP claims, TWI watershed), "
        f"and {n_target} regression targets spanning health, socioeconomic, and environmental domains. "
        f"Includes three spatially-blocked evaluation protocols (imputation, extrapolation, "
        f"super-resolution) with fixed splits stratified by SVI quartile, urban/rural density, "
        f"and hospital access. v{version}."
    )

    stats = {"n_pruned": n_pruned, "n_added": n_added, "n_desc_backfilled": n_desc_added,
             "total_fields": len(main_rs["field"])}
    return updated, stats


def build_manifest(summary: dict, timestamp: str) -> dict:
    return {
        "version": "v24.002",
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
            "twi_watershed": summary["enrichment_counts"]["twi"],
        },
        "columns": summary["columns"],
        "file_size_mb": summary["size_mb"],
        "description": (
            "GeoRSCT ZCTA-level features and labels. "
            "v24.002 fixes Croissant schema drift: prunes ghost fields from prior dataset "
            "versions, adds field descriptions for all columns, and adds TWI watershed "
            "features (topographic wetness index + slope from USGS ScienceBase StreamCat). "
            "NOTE: spatial lags (lag_acs_*) are intentionally excluded — solvers compute "
            "them at runtime from zcta_adjacency.parquet to ensure reproducibility."
        ),
    }


def verify_sync(api: HfApi, local_path: Path, repo_id: str) -> None:
    """
    Download the just-uploaded parquet and Croissant from HF and verify they agree.
    Fails loudly if any column is missing from Croissant or any ghost field remains.
    """
    print("\n=== SYNC VERIFICATION ===")
    croissant_path = api.hf_hub_download(repo_id, "croissant.json",
                                         repo_type="dataset", force_download=True)
    with open(croissant_path) as f:
        cr = json.load(f)
    main_rs = next(r for r in cr["recordSet"] if r["name"] == "georsct-main")
    hf_fields = {f["name"] for f in main_rs["field"]}

    hf_parquet = api.hf_hub_download(repo_id, "georsct_table.parquet",
                                     repo_type="dataset", force_download=True)
    hf_cols = set(pd.read_parquet(hf_parquet).columns)

    missing_from_croissant = hf_cols - hf_fields
    ghost_in_croissant     = hf_fields - hf_cols

    ok = True
    if missing_from_croissant:
        print(f"  FAIL: {len(missing_from_croissant)} parquet cols missing from Croissant:")
        for c in sorted(missing_from_croissant):
            print(f"    - {c}")
        ok = False
    if ghost_in_croissant:
        print(f"  FAIL: {len(ghost_in_croissant)} ghost fields in Croissant (not in parquet):")
        for c in sorted(ghost_in_croissant):
            print(f"    - {c}")
        ok = False

    if ok:
        print(f"  OK — HF parquet ({len(hf_cols)} cols) and Croissant ({len(hf_fields)} fields) are in sync.")
    else:
        print("  SYNC FAILED — investigate before next release.")
        sys.exit(1)


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
    hf_token = get_credential("HF_TOKEN")
    api = HfApi(token=hf_token)

    # Fetch and reconcile Croissant (prune ghosts + add new + backfill descriptions)
    print("  Reconciling croissant.json...")
    try:
        croissant_path = api.hf_hub_download(HF_REPO_ID, "croissant.json",
                                              repo_type="dataset")
        with open(croissant_path) as f:
            existing_croissant = json.load(f)
        df_tmp = pd.read_parquet(output_path)
        df_dtypes = {c: str(df_tmp[c].dtype) for c in df_tmp.columns}
        updated_croissant, stats = reconcile_croissant(
            existing_croissant, list(df_tmp.columns), df_dtypes,
            timestamp, "v24.002"
        )
        print(f"  Croissant: {stats['n_pruned']} pruned | {stats['n_added']} added | "
              f"{stats['n_desc_backfilled']} descriptions backfilled | "
              f"{stats['total_fields']} total fields")
        croissant_bytes = json.dumps(updated_croissant, indent=2).encode()
        api.upload_file(
            path_or_fileobj=croissant_bytes,
            path_in_repo="croissant.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="fix(croissant): reconcile schema to parquet — prune ghosts, add TWI descriptions, v24.002",
        )
        print(f"  -> {HF_REPO_ID}/croissant.json")
    except Exception as exc:
        print(f"  ERROR: Croissant reconciliation failed: {exc}")
        raise

    # Main table parquet
    print(f"  Uploading georsct_table.parquet ({summary['size_mb']} MB)...")
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="georsct_table.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="feat(v24.002): TWI watershed features + Croissant schema reconciliation",
    )
    print(f"  -> {HF_REPO_ID}/georsct_table.parquet")

    # Updated build manifest
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    api.upload_file(
        path_or_fileobj=manifest_bytes,
        path_in_repo="build_manifest.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="chore(manifest): update v24.002 column list and enrichment layers",
    )
    print(f"  -> {HF_REPO_ID}/build_manifest.json")

    # Post-upload sync verification
    verify_sync(api, output_path, HF_REPO_ID)

    print(f"\nDone. Dataset: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
