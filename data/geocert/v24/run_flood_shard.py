#!/usr/bin/env python3
"""
Sharded FEMA flood zone overlay: processes a SUBSET of counties and writes
per-county parquet results to S3 immediately after each county completes.

Part of a sharded pipeline where multiple SageMaker instances each process
different counties in parallel. The launcher assigns county lists via
--county-list JSON.

Per-county output:
  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/
      flood_county_areas/{fips}.parquet

Schema: zcta_id (str), zone_a_area_m2 (float64),
        zone_x500_area_m2 (float64), n_features (int)
"""

import argparse
import io
import json
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

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
S3_OUTPUT_PREFIX = (
    "rsct_curriculum/series_018/processed/flood_county_areas/"
)
S3_SUMMARY_PREFIX = (
    "rsct_curriculum/series_018/processed/flood_shard_summaries/"
)

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _s3_client():
    """Create a new boto3 S3 client (one per thread)."""
    import boto3
    return boto3.client("s3")


def _s3_upload_bytes(s3, buf: bytes, key: str) -> None:
    """Upload raw bytes to S3."""
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf)


def _s3_upload_file(s3, local_path: str, key: str) -> None:
    """Upload a local file to S3."""
    s3.upload_file(local_path, S3_BUCKET, key)


# ---------------------------------------------------------------------------
# Geometry helpers (copied from run_flood_combined.py)
# ---------------------------------------------------------------------------
def _signed_area(ring: list) -> float:
    """Compute signed area of a coordinate ring (shoelace formula)."""
    n = len(ring)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return area / 2.0


def _geom_to_rings(geom: dict) -> list:
    """Convert fiona/GeoJSON geometry to Esri-style rings."""
    if geom is None:
        return []
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        return [list(ring) for ring in coords]
    elif gtype == "MultiPolygon":
        rings = []
        for polygon_coords in coords:
            for ring in polygon_coords:
                rings.append(list(ring))
        return rings
    return []


def _rings_to_shapely_checked(rings: list, vstats: dict):
    """Convert Esri-style rings to shapely polygon with validation.

    Tracks Check 6 (area conservation on repair) via vstats dict.
    """
    if not rings:
        return None
    try:
        shells, holes = [], []
        for ring in rings:
            if len(ring) < 4:
                continue
            sa = _signed_area(ring)
            if sa < 0:
                shells.append(ring)
            elif sa > 0:
                holes.append(ring)
            else:
                shells.append(ring)

        if not shells:
            if holes:
                shells = holes
                holes = []
            else:
                return None

        if len(shells) == 1:
            poly = Polygon(shells[0], holes) if holes else Polygon(shells[0])
        else:
            polys = [Polygon(s) for s in shells]
            poly = MultiPolygon(
                [(p.exterior.coords, []) for p in polys if p.is_valid]
            )
            if poly.is_empty:
                poly = (
                    Polygon(shells[0], holes) if holes else Polygon(shells[0])
                )

        if not poly.is_valid:
            original_area = poly.area
            poly = poly.buffer(0)
            vstats["repaired"] += 1
            if original_area > 0:
                ratio = poly.area / original_area
                if ratio < 0.9 or ratio > 1.1:
                    vstats["area_ratio_bad"] += 1

        return poly
    except Exception:
        return None


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    """Classify FEMA flood zone to A, X500, or X.

    Args:
        fld_zone: FLD_ZONE field value from NFHL.
        zone_subty: ZONE_SUBTY field value from NFHL.

    Returns:
        One of "A", "X500", or "X".
    """
    if not fld_zone:
        return "X"
    zone = str(fld_zone).strip().upper()
    subty = str(zone_subty).strip().upper() if zone_subty else ""
    if zone in ("A", "AE", "AH", "AO", "AR", "A99", "VE", "V"):
        return "A"
    if "FLOODWAY" in zone:
        return "A"
    if zone == "X" and ("500" in subty or "0.2" in subty or "SHADED" in subty):
        return "X500"
    if "0.2 PCT" in zone:
        return "X500"
    return "X"


# ---------------------------------------------------------------------------
# Per-county processing
# ---------------------------------------------------------------------------
def _read_flood_features(data_path: str, layer_arg: dict) -> tuple:
    """Read flood features and classify into zone A / X500 polygon lists.

    Args:
        data_path: Path to .shp or .gdb file.
        layer_arg: Dict with optional 'layer' key for .gdb files.

    Returns:
        Tuple of (zone_a_polys, zone_x500_polys, n_features, vstats).
    """
    import fiona

    vstats = {
        "null_geom": 0, "degenerate": 0, "unclosed": 0,
        "cw_outer": 0, "ccw_outer": 0, "holes": 0,
        "repaired": 0, "area_ratio_bad": 0,
    }
    zone_a_polys = []
    zone_x500_polys = []
    n_features = 0

    with fiona.open(data_path, **layer_arg) as src:
        for feat in src:
            geom = feat.get("geometry", {})
            rings = _geom_to_rings(geom)
            if not rings:
                vstats["null_geom"] += 1
                continue
            n_features += 1

            # Check 4: winding order (sample first ring)
            if rings and len(rings[0]) >= 4:
                sa = _signed_area(rings[0])
                if sa < 0:
                    vstats["cw_outer"] += 1
                elif sa > 0:
                    vstats["ccw_outer"] += 1
            for i, ring in enumerate(rings):
                if len(ring) < 4:
                    vstats["degenerate"] += 1
                elif ring[0] != ring[-1]:
                    vstats["unclosed"] += 1
                if i > 0:
                    vstats["holes"] += 1

            props = dict(feat.get("properties", {}))
            zclass = classify_zone(
                props.get("FLD_ZONE", ""),
                props.get("ZONE_SUBTY", ""),
            )
            if zclass not in ("A", "X500"):
                continue

            poly = _rings_to_shapely_checked(rings, vstats)
            if poly is None:
                continue
            (zone_a_polys if zclass == "A" else zone_x500_polys).append(poly)

    return zone_a_polys, zone_x500_polys, n_features, vstats


def _overlay_zones(
    zone_a_polys: list,
    zone_x500_polys: list,
    county_zcta_ids: list,
    zcta_proj: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Intersect zone polygons with ZCTA boundaries, return per-ZCTA areas.

    Args:
        zone_a_polys: List of shapely polygons for Zone A.
        zone_x500_polys: List of shapely polygons for Zone X500.
        county_zcta_ids: List of zcta_id strings in this county.
        zcta_proj: GeoDataFrame of ZCTAs projected to EPSG:5070.

    Returns:
        DataFrame with columns zcta_id, zone_a_area_m2, zone_x500_area_m2.
        Only rows with nonzero area are included (sparse).
    """
    rows = []
    zcta_subset = zcta_proj[zcta_proj["zcta_id"].isin(county_zcta_ids)]

    for polys, col in [(zone_a_polys, "zone_a"), (zone_x500_polys, "zone_x500")]:
        if not polys:
            continue
        try:
            zone_gdf = gpd.GeoDataFrame(
                geometry=polys, crs="EPSG:4326"
            ).to_crs("EPSG:5070")
            if hasattr(zone_gdf.geometry, "union_all"):
                zone_union = zone_gdf.geometry.union_all()
            else:
                zone_union = zone_gdf.geometry.unary_union

            for _, zcta_row in zcta_subset.iterrows():
                intersection = zcta_row.geometry.intersection(zone_union)
                if not intersection.is_empty and intersection.area > 0:
                    rows.append({
                        "zcta_id": zcta_row["zcta_id"],
                        f"{col}_area_m2": intersection.area,
                    })
        except Exception as e:
            log.warning("  Overlay error: %s", e)

    if not rows:
        return pd.DataFrame(
            columns=["zcta_id", "zone_a_area_m2", "zone_x500_area_m2"]
        )

    df = pd.DataFrame(rows)
    # Aggregate: a ZCTA may appear in both zone_a and zone_x500 rows
    agg = df.groupby("zcta_id", as_index=False).sum()
    for c in ["zone_a_area_m2", "zone_x500_area_m2"]:
        if c not in agg.columns:
            agg[c] = 0.0
    return agg[["zcta_id", "zone_a_area_m2", "zone_x500_area_m2"]]


def process_county(
    county_info: dict,
    zcta_proj: gpd.GeoDataFrame,
    county_zcta_ids: list,
    work_dir: Path,
) -> dict:
    """Download zip, extract flood shapefile, overlay ZCTAs, upload result.

    Each invocation creates its own S3 client (boto3 is not thread-safe
    for a single client instance).

    Args:
        county_info: Dict with keys fips, key, dfirm_id.
        zcta_proj: ZCTA GeoDataFrame projected to EPSG:5070.
        county_zcta_ids: List of zcta_id strings in this county.
        work_dir: Temporary working directory.

    Returns:
        Dict with fips, n_features, n_rows, vstats, uploaded, error.
    """
    import threading
    fips = county_info["fips"]
    dfirm_id = county_info["dfirm_id"]
    zip_key = county_info["key"]
    tid = threading.get_ident()

    zip_path = work_dir / f"{dfirm_id}_{tid}.zip"
    extract_dir = work_dir / f"{dfirm_id}_{tid}_ext"

    empty = {
        "fips": fips, "n_features": 0, "n_rows": 0,
        "vstats": {}, "uploaded": False, "error": None,
    }

    s3 = _s3_client()
    try:
        s3.download_file(S3_BUCKET, zip_key, str(zip_path))

        if not zipfile.is_zipfile(zip_path):
            return {**empty, "error": "corrupt zip"}
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return {**empty, "error": f"bad entry: {bad}"}
            zf.extractall(extract_dir)

        # Find shapefile or geodatabase
        import fiona
        shp_paths = [
            p for p in extract_dir.rglob("*.shp")
            if p.stem.upper() == "S_FLD_HAZ_AR"
        ]
        gdb_paths = list(extract_dir.rglob("*.gdb"))

        if shp_paths:
            data_path = str(shp_paths[0])
            layer_arg = {}
        elif gdb_paths:
            data_path = str(gdb_paths[0])
            layers = fiona.listlayers(data_path)
            if "S_Fld_Haz_Ar" not in layers:
                return {**empty, "error": "No S_Fld_Haz_Ar layer"}
            layer_arg = {"layer": "S_Fld_Haz_Ar"}
        else:
            return {**empty, "error": "No .shp or .gdb found"}

        # Read and classify features
        zone_a, zone_x500, n_feat, vstats = _read_flood_features(
            data_path, layer_arg
        )

        # Overlay against ZCTAs
        df = _overlay_zones(zone_a, zone_x500, county_zcta_ids, zcta_proj)
        df["n_features"] = n_feat

        # Ensure correct dtypes
        df["zcta_id"] = df["zcta_id"].astype(str)
        df["zone_a_area_m2"] = df["zone_a_area_m2"].astype("float64")
        df["zone_x500_area_m2"] = df["zone_x500_area_m2"].astype("float64")
        df["n_features"] = df["n_features"].astype(int)

        # Filter to nonzero rows only (sparse output)
        mask = (df["zone_a_area_m2"] > 0) | (df["zone_x500_area_m2"] > 0)
        df = df[mask].reset_index(drop=True)

        # Upload per-county parquet to S3 immediately
        uploaded = False
        if len(df) > 0:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            s3_key = f"{S3_OUTPUT_PREFIX}{fips}.parquet"
            _s3_upload_bytes(s3, buf.getvalue(), s3_key)
            uploaded = True

        return {
            "fips": fips,
            "n_features": n_feat,
            "n_rows": len(df),
            "vstats": vstats,
            "uploaded": uploaded,
            "error": None,
        }

    except Exception as e:
        return {**empty, "error": str(e)}
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------
def _log_progress(
    done: int,
    total: int,
    t_start: float,
    failed: int,
    total_features: int,
) -> None:
    """Log progress at regular intervals."""
    elapsed = time.time() - t_start
    rate = done / (elapsed / 60) if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    log.info(
        "  [%d/%d] %.0f/min, ETA %.0f min, %d failed, %d features",
        done, total, rate, eta, failed, total_features,
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for sharded flood zone processing."""
    parser = argparse.ArgumentParser(
        description="Sharded FEMA flood zone overlay"
    )
    parser.add_argument(
        "--county-list", required=True,
        help="Path to JSON file with list of county dicts",
    )
    parser.add_argument(
        "--tiger-dir",
        default="/opt/ml/processing/input/tiger",
        help="TIGER shapefile directory (ZCTA subset for this shard)",
    )
    parser.add_argument(
        "--data-dir",
        default="/opt/ml/processing/input/data",
        help="Directory containing zcta_county_crosswalk.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
        help="Local output directory (summary only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="nfhl_shard_"))

    t_job_start = time.time()

    # =================================================================
    # 1. Load county list from JSON
    # =================================================================
    log.info("=== LOADING COUNTY LIST ===")
    with open(args.county_list) as f:
        county_list = json.load(f)
    log.info("  %d counties in this shard", len(county_list))

    # =================================================================
    # 2. Load TIGER + crosswalk, project to EPSG:5070
    # =================================================================
    log.info("=== LOADING TIGER ===")
    tiger_path = Path(args.tiger_dir)
    shp_files = list(tiger_path.glob("*.shp"))
    if not shp_files:
        log.error("No .shp files found in %s", tiger_path)
        sys.exit(1)

    zcta_geo = gpd.read_file(shp_files[0], engine="pyogrio")
    zcta_geo = zcta_geo[["ZCTA5CE20", "geometry"]].rename(
        columns={"ZCTA5CE20": "zcta_id"}
    )
    zcta_geo["zcta_id"] = zcta_geo["zcta_id"].astype(str).str.zfill(5)
    zcta_geo = zcta_geo.to_crs("EPSG:4326")
    log.info("  %d ZCTAs loaded", len(zcta_geo))

    xwalk = pd.read_parquet(
        Path(args.data_dir) / "zcta_county_crosswalk.parquet"
    )
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    zcta_counties = dict(
        zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str))
    )
    zcta_geo["county_fips"] = (
        zcta_geo["zcta_id"].map(zcta_counties).fillna("unknown")
    )

    log.info("  Projecting to EPSG:5070...")
    zcta_proj = zcta_geo.to_crs("EPSG:5070")
    log.info("  Projection complete: %d ZCTAs", len(zcta_proj))

    # =================================================================
    # 3. Build county -> ZCTA ID mapping
    # =================================================================
    log.info("=== BUILDING COUNTY-ZCTA MAP ===")
    county_zcta_map = {}
    for county_fips, group in zcta_geo.groupby("county_fips"):
        county_zcta_map[county_fips] = list(group["zcta_id"])

    to_process = []
    for entry in county_list:
        fips = entry["fips"]
        zcta_ids = county_zcta_map.get(fips, [])
        if not zcta_ids:
            log.warning("  No ZCTAs for county %s -- skipping", fips)
            continue
        to_process.append({
            "county_info": {
                "fips": fips,
                "key": entry["key"],
                "dfirm_id": entry["dfirm_id"],
            },
            "zcta_ids": zcta_ids,
        })
    log.info("  %d counties matched to ZCTAs", len(to_process))

    # =================================================================
    # 4. Process counties in parallel
    # =================================================================
    n_cpus = multiprocessing.cpu_count()
    n_threads = n_cpus
    log.info(
        "=== PROCESSING %d COUNTIES (%d threads, %d CPUs) ===",
        len(to_process), n_threads, n_cpus,
    )

    t_start = time.time()
    done = 0
    failed = 0
    total_features = 0
    total_uploaded = 0

    agg_vstats = {
        "null_geom": 0, "degenerate": 0, "unclosed": 0,
        "cw_outer": 0, "ccw_outer": 0, "holes": 0,
        "repaired": 0, "area_ratio_bad": 0,
    }
    vstats_lock = Lock()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(
                process_county,
                item["county_info"],
                zcta_proj,
                item["zcta_ids"],
                work_dir,
            ): item["county_info"]["fips"]
            for item in to_process
        }

        for fut in as_completed(futures):
            fips = futures[fut]
            try:
                result = fut.result()
                done += 1

                if result["error"]:
                    log.warning("  SKIP %s: %s", fips, result["error"])
                    failed += 1
                else:
                    total_features += result["n_features"]
                    if result["uploaded"]:
                        total_uploaded += 1

                    with vstats_lock:
                        for k in agg_vstats:
                            agg_vstats[k] += result.get("vstats", {}).get(
                                k, 0
                            )

                if done % 25 == 0 or done == len(to_process):
                    _log_progress(
                        done, len(to_process), t_start, failed,
                        total_features,
                    )

            except Exception as e:
                log.warning("  ERROR %s: %s", fips, e)
                failed += 1
                done += 1

    elapsed_total = time.time() - t_job_start
    elapsed_processing = time.time() - t_start

    # =================================================================
    # 5. Validation summary
    # =================================================================
    log.info("=== VALIDATION ===")

    log.info("  Check 3 - Geometry:")
    log.info(
        "    null_geom=%d  degenerate_rings=%d  unclosed_rings=%d",
        agg_vstats["null_geom"],
        agg_vstats["degenerate"],
        agg_vstats["unclosed"],
    )

    log.info("  Check 4 - Winding:")
    log.info(
        "    CW outer (Esri-correct)=%d  CCW outer (reversed)=%d  holes=%d",
        agg_vstats["cw_outer"],
        agg_vstats["ccw_outer"],
        agg_vstats["holes"],
    )
    cw_total = agg_vstats["cw_outer"] + agg_vstats["ccw_outer"]
    if cw_total > 0:
        cw_pct = agg_vstats["cw_outer"] / cw_total * 100
        log.info(
            "    %.1f%% CW (expected: >95%% for Esri shapefiles)", cw_pct
        )

    log.info("  Check 6 - Area conservation:")
    log.info(
        "    repaired=%d  area_ratio_bad (>10%% change)=%d",
        agg_vstats["repaired"],
        agg_vstats["area_ratio_bad"],
    )
    if agg_vstats["repaired"] > 0:
        bad_pct = agg_vstats["area_ratio_bad"] / agg_vstats["repaired"] * 100
        log.info("    %.1f%% of repairs exceeded 10%% area change", bad_pct)

    # =================================================================
    # 6. Write summary JSON (local + S3)
    # =================================================================
    instance_type = os.environ.get(
        "INSTANCE_TYPE",
        os.environ.get("SM_RESOURCE_CONFIG_INSTANCE_TYPE", "unknown"),
    )
    shard_id = os.environ.get("SHARD_ID", "unknown")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shard_id": shard_id,
        "counties_total": len(county_list),
        "counties_matched": len(to_process),
        "counties_processed": done - failed,
        "counties_failed": failed,
        "counties_uploaded": total_uploaded,
        "total_features": total_features,
        "compute": {
            "instance_type": instance_type,
            "n_cpus": n_cpus,
            "n_threads": n_threads,
            "processing_elapsed_sec": round(elapsed_processing, 1),
            "processing_elapsed_min": round(elapsed_processing / 60, 1),
            "total_elapsed_sec": round(elapsed_total, 1),
            "total_elapsed_min": round(elapsed_total / 60, 1),
        },
        "validation": {
            "check_3_geometry": {
                "null_geom": agg_vstats["null_geom"],
                "degenerate_rings": agg_vstats["degenerate"],
                "unclosed_rings": agg_vstats["unclosed"],
            },
            "check_4_winding": {
                "cw_outer": agg_vstats["cw_outer"],
                "ccw_outer": agg_vstats["ccw_outer"],
                "holes": agg_vstats["holes"],
            },
            "check_6_area_conservation": {
                "repaired": agg_vstats["repaired"],
                "area_ratio_bad": agg_vstats["area_ratio_bad"],
            },
        },
    }

    # Local summary
    summary_path = output_dir / "shard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved local summary: %s", summary_path)

    # S3 summary
    try:
        s3 = _s3_client()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        s3_summary_key = f"{S3_SUMMARY_PREFIX}{shard_id}_{ts}.json"
        _s3_upload_bytes(
            s3, json.dumps(summary, indent=2).encode(), s3_summary_key
        )
        log.info("Uploaded summary: s3://%s/%s", S3_BUCKET, s3_summary_key)
    except Exception as e:
        log.warning("Failed to upload summary to S3: %s", e)

    # =================================================================
    # 7. Final log
    # =================================================================
    log.info("=== COMPLETE ===")
    log.info(
        "  Counties: %d processed, %d failed, %d uploaded",
        done - failed, failed, total_uploaded,
    )
    log.info("  Features: %d total", total_features)
    log.info("  Processing time: %.1f min", elapsed_processing / 60)
    log.info("  Total time: %.1f min", elapsed_total / 60)

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
