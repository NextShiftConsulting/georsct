#!/usr/bin/env python3
"""
Merge per-county flood area parquets into final flood_zones_zcta.parquet.

Reads sharded county parquets from S3 (one per FIPS), aggregates by ZCTA,
joins against TIGER ZCTA boundaries to compute flood zone percentages,
and uploads the merged result.

Can run locally (--local) or as a SageMaker processing job.

Instance: ml.m5.xlarge (4 vCPU, 16 GB -- merge is lightweight)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

S3_BUCKET = "swarm-yrsn-datasets"
S3_COUNTY_PREFIX = (
    "rsct_curriculum/series_018/processed/flood_county_areas/"
)
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
S3_PROVENANCE_KEY = (
    "rsct_curriculum/series_018/processed/flood_zones_provenance.json"
)

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3_client(local: bool):
    """Create S3 client, using nsc-swarm profile when running locally."""
    import boto3
from swarm_auth import get_aws_credentials
    if local:
        _aws = get_aws_credentials()
        return boto3.client("s3", **_aws)
    return boto3.client("s3")


def _s3_list_parquets(s3) -> List[Tuple[str, str]]:
    """List all parquet files under the county areas prefix.

    Returns:
        List of (s3_key, fips_code) tuples.
    """
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_COUNTY_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if not fname.endswith(".parquet"):
                continue
            fips = fname.replace(".parquet", "")
            results.append((key, fips))
    return results


def _s3_upload(s3, local_path: str, key: str) -> None:
    """Upload a local file to S3."""
    try:
        s3.upload_file(local_path, S3_BUCKET, key)
        log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


# ---------------------------------------------------------------------------
# Step 1-2: Download and concat county parquets
# ---------------------------------------------------------------------------

def download_county_parquets(
    s3, work_dir: Path
) -> Tuple[pd.DataFrame, int, int]:
    """Download all county parquets from S3 and concatenate.

    Args:
        s3: Boto3 S3 client.
        work_dir: Local directory for temporary downloads.

    Returns:
        Tuple of (concatenated DataFrame, n_county_files, total_features).
    """
    parquet_list = _s3_list_parquets(s3)
    log.info("Found %d county parquet files on S3", len(parquet_list))

    if not parquet_list:
        raise RuntimeError("No county parquets found under " + S3_COUNTY_PREFIX)

    dl_dir = work_dir / "county_parquets"
    dl_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total_features = 0
    for i, (key, fips) in enumerate(parquet_list):
        local_path = dl_dir / f"{fips}.parquet"
        s3.download_file(S3_BUCKET, key, str(local_path))
        df = pd.read_parquet(local_path)
        total_features += int(df["n_features"].sum()) if "n_features" in df.columns else 0
        frames.append(df)
        if (i + 1) % 200 == 0:
            log.info("  downloaded %d / %d county files", i + 1, len(parquet_list))

    log.info("Downloaded all %d county files, concatenating", len(parquet_list))
    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Combined shape: %d rows, %d unique ZCTAs",
        len(combined),
        combined["zcta_id"].nunique(),
    )
    return combined, len(parquet_list), total_features


# ---------------------------------------------------------------------------
# Step 3: Group by ZCTA and sum areas
# ---------------------------------------------------------------------------

def aggregate_by_zcta(county_df: pd.DataFrame) -> pd.DataFrame:
    """Group by zcta_id and sum zone areas across counties.

    ZCTAs spanning county boundaries receive additive areas from
    multiple county parquets.

    Args:
        county_df: Concatenated county-level flood area data.

    Returns:
        DataFrame with one row per zcta_id.
    """
    agg = (
        county_df
        .groupby("zcta_id", as_index=False)
        .agg(
            zone_a_area_m2=("zone_a_area_m2", "sum"),
            zone_x500_area_m2=("zone_x500_area_m2", "sum"),
        )
    )
    log.info("Aggregated to %d unique ZCTAs with flood data", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Step 4: Load TIGER and compute ZCTA areas
# ---------------------------------------------------------------------------

def load_tiger_zcta_areas(
    tiger_dir: str, crosswalk_path: str
) -> pd.DataFrame:
    """Load TIGER ZCTA boundaries, filter to CONUS, compute areas.

    Args:
        tiger_dir: Path to TIGER ZCTA shapefile directory.
        crosswalk_path: Path to zcta_county_crosswalk.parquet.

    Returns:
        DataFrame with columns (zcta_id, zcta_area_m2).
    """
    log.info("Loading TIGER ZCTA boundaries from %s", tiger_dir)
    tiger_path = Path(tiger_dir)
    shp_files = list(tiger_path.glob("*.shp"))
    if shp_files:
        gdf = gpd.read_file(shp_files[0], engine="pyogrio")
    else:
        gdf = gpd.read_file(tiger_dir)
    log.info("  raw TIGER ZCTAs: %d", len(gdf))

    # Identify ZCTA column
    zcta_col = None
    for candidate in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10"):
        if candidate in gdf.columns:
            zcta_col = candidate
            break
    if zcta_col is None:
        raise ValueError(
            "Cannot find ZCTA ID column in TIGER. Columns: "
            + ", ".join(gdf.columns.tolist())
        )
    gdf = gdf.rename(columns={zcta_col: "zcta_id"})

    # Filter to CONUS using crosswalk (vectorized)
    log.info("Loading crosswalk from %s", crosswalk_path)
    xwalk = pd.read_parquet(crosswalk_path)
    fips_col = "county_fips" if "county_fips" in xwalk.columns else "GEOID"
    zcta_xwalk_col = "zcta_id" if "zcta_id" in xwalk.columns else "ZCTA5CE20"
    xwalk["_state_fips"] = xwalk[fips_col].astype(str).str[:2]
    conus_mask = ~xwalk["_state_fips"].isin(NON_CONUS_PREFIXES)
    conus_zctas = set(xwalk.loc[conus_mask, zcta_xwalk_col].astype(str).unique())

    gdf["zcta_id"] = gdf["zcta_id"].astype(str)
    gdf = gdf[gdf["zcta_id"].isin(conus_zctas)].copy()
    log.info("  CONUS ZCTAs after crosswalk filter: %d", len(gdf))

    # Project to EPSG:5070 and compute area
    gdf = gdf.to_crs(epsg=5070)
    areas = pd.DataFrame({
        "zcta_id": gdf["zcta_id"].values,
        "zcta_area_m2": gdf.geometry.area.values,
    })
    log.info(
        "  ZCTA area stats: min=%.0f max=%.0f mean=%.0f m2",
        areas["zcta_area_m2"].min(),
        areas["zcta_area_m2"].max(),
        areas["zcta_area_m2"].mean(),
    )
    return areas


# ---------------------------------------------------------------------------
# Step 5-6: Compute percentages and flags
# ---------------------------------------------------------------------------

def compute_flood_percentages(
    zcta_areas: pd.DataFrame, flood_agg: pd.DataFrame
) -> pd.DataFrame:
    """Left join full ZCTA list to flood areas, compute percentages.

    All CONUS ZCTAs appear in output. ZCTAs with no flood data get
    0/0/100 for zone_a/zone_x500/zone_x percentages.

    Args:
        zcta_areas: Full CONUS ZCTA list with zcta_area_m2.
        flood_agg: Aggregated flood areas by zcta_id.

    Returns:
        Final DataFrame with flood zone percentages and flags.
    """
    merged = zcta_areas.merge(flood_agg, on="zcta_id", how="left")
    merged["zone_a_area_m2"] = merged["zone_a_area_m2"].fillna(0.0)
    merged["zone_x500_area_m2"] = merged["zone_x500_area_m2"].fillna(0.0)

    # Percentages capped at 100
    merged["flood_pct_zone_a"] = np.minimum(
        merged["zone_a_area_m2"] / merged["zcta_area_m2"] * 100.0, 100.0
    )
    merged["flood_pct_zone_x500"] = np.minimum(
        merged["zone_x500_area_m2"] / merged["zcta_area_m2"] * 100.0, 100.0
    )
    merged["flood_pct_zone_x"] = np.maximum(
        100.0 - merged["flood_pct_zone_a"] - merged["flood_pct_zone_x500"],
        0.0,
    )

    # Boolean SFHA flag
    merged["flood_sfha"] = merged["flood_pct_zone_a"] > 0.0

    log.info(
        "Flood percentages computed for %d ZCTAs, %d with SFHA",
        len(merged),
        merged["flood_sfha"].sum(),
    )
    return merged


# ---------------------------------------------------------------------------
# Step 7: Validation checks
# ---------------------------------------------------------------------------

def run_check_5(
    result: pd.DataFrame, crosswalk_path: str
) -> Dict:
    """Check 5: Harris County (FIPS 48201) ZCTAs have Zone A coverage.

    Args:
        result: Final merged DataFrame.
        crosswalk_path: Path to zcta_county_crosswalk.parquet.

    Returns:
        Dict with check results.
    """
    xwalk = pd.read_parquet(crosswalk_path)
    fips_col = "county_fips" if "county_fips" in xwalk.columns else "GEOID"
    zcta_col = "zcta_id" if "zcta_id" in xwalk.columns else "ZCTA5CE20"

    harris_zctas = set(
        xwalk.loc[xwalk[fips_col].astype(str) == "48201", zcta_col]
        .astype(str)
        .unique()
    )
    harris_rows = result[result["zcta_id"].isin(harris_zctas)]
    n_harris = len(harris_rows)
    n_with_a = int((harris_rows["flood_pct_zone_a"] > 0).sum())
    pct_with_a = (n_with_a / n_harris * 100) if n_harris > 0 else 0.0

    passed = n_with_a > 0
    log.info(
        "Check 5 (Harris County): %d ZCTAs, %d with Zone A (%.1f%%) -- %s",
        n_harris, n_with_a, pct_with_a,
        "PASS" if passed else "FAIL",
    )
    return {
        "check": 5,
        "name": "harris_county_zone_a",
        "passed": passed,
        "n_harris_zctas": n_harris,
        "n_with_zone_a": n_with_a,
        "pct_with_zone_a": round(pct_with_a, 2),
    }


def run_check_7(result: pd.DataFrame) -> Dict:
    """Check 7: Distribution stats and overflow check.

    Reports min, max, mean, median, zero% for each zone column.
    Flags overflow where zone_a + zone_x500 > 100%.

    Args:
        result: Final merged DataFrame.

    Returns:
        Dict with check results.
    """
    stats = {}
    for col in ("flood_pct_zone_a", "flood_pct_zone_x500", "flood_pct_zone_x"):
        vals = result[col]
        n_zero = int((vals == 0.0).sum())
        stats[col] = {
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "zero_pct": round(n_zero / len(vals) * 100, 2),
        }
        log.info(
            "Check 7 [%s]: min=%.4f max=%.4f mean=%.4f median=%.4f zero=%.1f%%",
            col,
            stats[col]["min"],
            stats[col]["max"],
            stats[col]["mean"],
            stats[col]["median"],
            stats[col]["zero_pct"],
        )

    # Overflow check
    overflow = result["flood_pct_zone_a"] + result["flood_pct_zone_x500"]
    n_overflow = int((overflow > 100.0).sum())
    max_overflow = round(float(overflow.max()), 4)
    log.info(
        "Check 7 overflow: %d ZCTAs with zone_a + zone_x500 > 100%%, max=%.2f%%",
        n_overflow, max_overflow,
    )

    return {
        "check": 7,
        "name": "distribution_stats",
        "passed": True,
        "column_stats": stats,
        "overflow_count": n_overflow,
        "overflow_max_pct": max_overflow,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def build_provenance(
    n_county_files: int,
    total_features: int,
    result: pd.DataFrame,
    check5: Dict,
    check7: Dict,
    elapsed_s: float,
) -> Dict:
    """Build provenance JSON for the merge run.

    Args:
        n_county_files: Number of county parquets processed.
        total_features: Sum of n_features across all counties.
        result: Final merged DataFrame.
        check5: Check 5 results dict.
        check7: Check 7 results dict.
        elapsed_s: Wall-clock seconds elapsed.

    Returns:
        Provenance dict ready for JSON serialization.
    """
    n_zctas = len(result)
    n_with_a = int((result["flood_pct_zone_a"] > 0).sum())
    zone_a_coverage_pct = round(n_with_a / n_zctas * 100, 2) if n_zctas else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "per-county flood area parquets from sharded overlay",
        "method": "sharded_extract_overlay + merge",
        "n_county_files": n_county_files,
        "n_counties_total": n_county_files,
        "total_features": total_features,
        "n_zctas": n_zctas,
        "zone_a_coverage_pct": zone_a_coverage_pct,
        "validation": {
            "check_5": check5,
            "check_7": check7,
        },
        "compute": {
            "elapsed_seconds": round(elapsed_s, 1),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for flood merge pipeline."""
    parser = argparse.ArgumentParser(
        description="Merge per-county flood parquets into flood_zones_zcta.parquet"
    )
    parser.add_argument(
        "--tiger-dir",
        required=True,
        help="Path to TIGER ZCTA shapefile directory (full CONUS)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing zcta_county_crosswalk.parquet",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local output directory for parquet and provenance JSON",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use AWS profile nsc-swarm instead of IAM role",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== Flood merge started ===")
    log.info("  tiger-dir:  %s", args.tiger_dir)
    log.info("  data-dir:   %s", args.data_dir)
    log.info("  output-dir: %s", args.output_dir)
    log.info("  local:      %s", args.local)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    crosswalk_path = str(Path(args.data_dir) / "zcta_county_crosswalk.parquet")

    # S3 client
    s3 = _make_s3_client(local=args.local)

    # Step 1-2: download and concat
    log.info("--- Step 1-2: download and concat county parquets ---")
    county_df, n_county_files, total_features = download_county_parquets(
        s3, work_dir
    )

    # Step 3: aggregate by ZCTA
    log.info("--- Step 3: aggregate by ZCTA ---")
    flood_agg = aggregate_by_zcta(county_df)

    # Step 4: TIGER ZCTA areas
    log.info("--- Step 4: load TIGER and compute ZCTA areas ---")
    zcta_areas = load_tiger_zcta_areas(args.tiger_dir, crosswalk_path)

    # Step 5-6: compute percentages
    log.info("--- Step 5-6: compute flood percentages ---")
    result = compute_flood_percentages(zcta_areas, flood_agg)

    # Step 7: validation
    log.info("--- Step 7: validation checks ---")
    check5 = run_check_5(result, crosswalk_path)
    check7 = run_check_7(result)

    # Select output columns
    output_cols = [
        "zcta_id",
        "zcta_area_m2",
        "zone_a_area_m2",
        "zone_x500_area_m2",
        "flood_pct_zone_a",
        "flood_pct_zone_x500",
        "flood_pct_zone_x",
        "flood_sfha",
    ]
    result = result[output_cols].copy()

    # Save locally
    out_parquet = output_dir / "flood_zones_zcta.parquet"
    result.to_parquet(str(out_parquet), index=False)
    log.info("Saved %s (%d rows)", out_parquet, len(result))

    elapsed = time.time() - t0
    provenance = build_provenance(
        n_county_files, total_features, result, check5, check7, elapsed
    )
    out_prov = output_dir / "flood_zones_provenance.json"
    with open(str(out_prov), "w") as f:
        json.dump(provenance, f, indent=2)
    log.info("Saved %s", out_prov)

    # Upload to S3
    log.info("--- Uploading to S3 ---")
    _s3_upload(s3, str(out_parquet), S3_OUTPUT_KEY)
    _s3_upload(s3, str(out_prov), S3_PROVENANCE_KEY)

    log.info("=== Flood merge complete in %.1f seconds ===", elapsed)
    log.info("  ZCTAs: %d, SFHA: %d", len(result), result["flood_sfha"].sum())


if __name__ == "__main__":
    main()
