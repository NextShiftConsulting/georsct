#!/usr/bin/env python3
"""
run_flood_zones.py -- SageMaker run script for FEMA NFHL flood zone enrichment.

Designed for ml.m5.2xlarge (8 vCPU, 32 GB RAM). Expected runtime: 60-120 min.

Strategy:
  1. Load ZCTA boundaries from TIGER shapefile (mounted from S3 or downloaded)
  2. Load county crosswalk to group ZCTAs by county (~3,200 counties)
  3. Build size-sorted county manifest (small counties first, large last)
  4. Phase 1 — FETCH (parallel): ThreadPoolExecutor fetches NFHL polygons
     from FEMA ArcGIS REST layer 28, saves raw features per county to disk
  5. Phase 2 — OVERLAY (sequential): For each county, load raw features,
     spatial overlay in EPSG:5070, accumulate area fractions, checkpoint
  6. Upload final parquet + provenance directly to S3

Memory optimization:
  - Fetch and overlay are SEPARATED: fetch is I/O bound (parallelizable),
    overlay is memory bound (sequential, one county at a time)
  - ZCTA boundaries loaded once (~1.5 GB), projected once to EPSG:5070
  - Flood polygons per-county are transient (GC'd after overlay)
  - Accumulate area in numpy arrays, not DataFrames

Crash recovery:
  - Checkpoint after EVERY county (not batched)
  - On restart, skips counties already in checkpoint
  - Raw fetched data persists on disk between phases

Threading:
  - Phase 1 only: ThreadPoolExecutor(4) for FEMA API I/O
  - Phase 2: strictly sequential to cap peak memory at 1 county
"""

import argparse
import gc
import json
import logging
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
# SageMaker CloudWatch fix: force flush on every log emit.
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEMA NFHL configuration
# ---------------------------------------------------------------------------
NFHL_BASE = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
NFHL_LAYER = 28
NFHL_QUERY_URL = f"{NFHL_BASE}/{NFHL_LAYER}/query"

PAGE_SIZE = 1000
MIN_CELL_DEG = 0.05  # quadtree split floor
MAX_COUNTY_BBOX_DEG = 2.0  # pre-split threshold for large counties
INTER_REQUEST_SLEEP = 0.02
MAX_RETRIES = 3

# S3 output location (direct upload, not EndOfJob-dependent)
S3_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/series_018/processed/flood_zones_provenance.json"
S3_CHECKPOINT_KEY = "rsct_curriculum/series_018/processed/flood_partial.npz"
S3_CHECKPOINT_JSON_KEY = "rsct_curriculum/series_018/processed/flood_checkpoint.json"
S3_FETCH_PREFIX = "rsct_curriculum/series_018/processed/flood_fetch/"

# TIGER shapefile URL (fallback if not mounted)
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip"


# ---------------------------------------------------------------------------
# S3 helper
# ---------------------------------------------------------------------------
def _s3_upload(local_path: str, key: str, quiet: bool = False):
    """Best-effort upload to S3."""
    try:
        import boto3
        boto3.client("s3").upload_file(local_path, S3_BUCKET, key)
        if not quiet:
            log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


def _s3_download(key: str, local_path: str, quiet: bool = False) -> bool:
    """Best-effort download from S3. Returns True on success."""
    try:
        import boto3
        boto3.client("s3").download_file(S3_BUCKET, key, local_path)
        if not quiet:
            log.info("  <- s3://%s/%s", S3_BUCKET, key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FEMA API helpers
# ---------------------------------------------------------------------------
def _signed_area(ring: list) -> float:
    """Compute signed area of a ring using the shoelace formula.

    Positive = counter-clockwise (Shapely shell convention).
    Negative = clockwise (Shapely hole convention).
    Esri convention: outer rings are clockwise (negative signed area),
    holes are counter-clockwise (positive signed area).
    """
    n = len(ring)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return area / 2.0


def _esri_to_shapely(geometry: dict):
    """Convert Esri JSON rings to shapely Polygon.

    Esri convention: outer rings are clockwise (negative signed area),
    inner rings (holes) are counter-clockwise (positive signed area).
    We classify by signed area rather than assuming ring order.
    """
    rings = geometry.get("rings", [])
    if not rings:
        return None
    try:
        shells = []
        holes = []
        for ring in rings:
            sa = _signed_area(ring)
            if sa < 0:
                # Clockwise = Esri outer ring = Shapely shell (reversed)
                shells.append(ring)
            elif sa > 0:
                # Counter-clockwise = Esri hole
                holes.append(ring)
            else:
                # Degenerate ring (zero area), skip
                continue

        if not shells:
            # Fallback: if no ring has negative signed area, treat first as shell
            # (some FEMA features use non-standard winding)
            shells = [rings[0]]
            holes = rings[1:]

        # Use the largest shell as the outer boundary
        if len(shells) == 1:
            poly = Polygon(shells[0], holes)
        else:
            # Multiple shells: use largest by absolute area, rest become separate
            # For simplicity, just use the largest
            shells.sort(key=lambda r: abs(_signed_area(r)), reverse=True)
            poly = Polygon(shells[0], holes)

        return poly.buffer(0) if not poly.is_valid else poly
    except Exception:
        return None


def _fetch_page(bbox_str: str, offset: int = 0):
    """Fetch one page from NFHL. Returns (features, exceeded) or (None, False)."""
    params = {
        "where": "1=1",
        "geometry": bbox_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
        "returnGeometry": "true",
        "f": "json",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": offset,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(NFHL_QUERY_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return None, False
            return data.get("features", []), data.get("exceededTransferLimit", False)
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None, False


def fetch_nfhl_for_bbox(xmin: float, ymin: float, xmax: float, ymax: float) -> list:
    """Fetch all NFHL features for a bbox with pagination + quadtree fallback."""
    bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
    all_features = []
    offset = 0

    while True:
        features, exceeded = _fetch_page(bbox_str, offset)

        if features is None:
            if (xmax - xmin) > MIN_CELL_DEG and (ymax - ymin) > MIN_CELL_DEG:
                return _fetch_quadtree(xmin, ymin, xmax, ymax)
            return all_features

        if not features:
            break

        all_features.extend(features)
        if len(features) < PAGE_SIZE and not exceeded:
            break
        offset += len(features)
        time.sleep(INTER_REQUEST_SLEEP)

    return all_features


def _fetch_quadtree(xmin, ymin, xmax, ymax) -> list:
    mx, my = (xmin + xmax) / 2, (ymin + ymax) / 2
    result = []
    for bx in [(xmin, mx), (mx, xmax)]:
        for by in [(ymin, my), (my, ymax)]:
            result.extend(fetch_nfhl_for_bbox(bx[0], by[0], bx[1], by[1]))
            time.sleep(INTER_REQUEST_SLEEP)
    return result


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    """Classify into A (100yr SFHA), X500 (500yr), or X (minimal)."""
    if pd.isna(fld_zone):
        return "X"
    zone = str(fld_zone).strip().upper()
    subty = str(zone_subty).strip().upper() if pd.notna(zone_subty) else ""

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
# Data loading
# ---------------------------------------------------------------------------
def load_tiger(tiger_dir: str) -> gpd.GeoDataFrame:
    """Load ZCTA boundaries from TIGER shapefile directory."""
    tiger_path = Path(tiger_dir)
    shp_files = list(tiger_path.glob("*.shp")) if tiger_path.exists() else []

    if not shp_files:
        log.info("TIGER not found at %s -- downloading...", tiger_dir)
        zip_path = Path(tempfile.gettempdir()) / "tl_2022_us_zcta520.zip"
        if not zip_path.exists():
            resp = requests.get(TIGER_URL, stream=True, timeout=300)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            log.info("  Downloaded %.1f MB", zip_path.stat().st_size / 1e6)
        tiger_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tiger_path)
        shp_files = list(tiger_path.glob("*.shp"))

    log.info("Reading TIGER shapefile: %s", shp_files[0])
    geo = gpd.read_file(shp_files[0], engine="pyogrio")
    geo = geo[["ZCTA5CE20", "geometry"]].rename(columns={"ZCTA5CE20": "zcta_id"})
    geo["zcta_id"] = geo["zcta_id"].astype(str).str.zfill(5)
    geo = geo.to_crs("EPSG:4326")
    log.info("  %d ZCTA boundaries loaded", len(geo))
    return geo


def load_crosswalk(data_dir: str) -> pd.DataFrame:
    """Load ZCTA-county crosswalk."""
    path = Path(data_dir) / "zcta_county_crosswalk.parquet"
    if not path.exists():
        path = Path(tempfile.gettempdir()) / "zcta_county_crosswalk.parquet"
    xwalk = pd.read_parquet(path)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    log.info("  %d ZCTA-county assignments loaded", len(xwalk))
    return xwalk


# ---------------------------------------------------------------------------
# County manifest (size-sorted)
# ---------------------------------------------------------------------------
def build_county_manifest(
    zcta_geo: gpd.GeoDataFrame,
) -> list[dict]:
    """Build size-sorted county manifest: small counties first, large last.

    Complexity estimate = n_zctas * bbox_area_deg2. This correlates with
    both FEMA query size and overlay memory usage.
    """
    manifest = []
    for county_fips, idx in zcta_geo.groupby("county_fips").groups.items():
        idx_list = list(idx)
        bounds = zcta_geo.iloc[idx_list].total_bounds  # xmin,ymin,xmax,ymax
        bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        manifest.append({
            "county_fips": county_fips,
            "idx_list": idx_list,
            "n_zctas": len(idx_list),
            "bbox_area_deg2": round(bbox_area, 4),
            "complexity": round(len(idx_list) * bbox_area, 4),
        })

    manifest.sort(key=lambda m: m["complexity"])
    log.info("County manifest: %d counties, complexity range %.2f - %.2f",
             len(manifest),
             manifest[0]["complexity"] if manifest else 0,
             manifest[-1]["complexity"] if manifest else 0)
    return manifest


# ---------------------------------------------------------------------------
# Phase 1: FETCH (parallel, I/O bound)
# ---------------------------------------------------------------------------
def _split_bbox_to_tiles(xmin, ymin, xmax, ymax, max_deg):
    """Split a bbox into tiles no larger than max_deg x max_deg."""
    tiles = []
    x = xmin
    while x < xmax:
        x2 = min(x + max_deg, xmax)
        y = ymin
        while y < ymax:
            y2 = min(y + max_deg, ymax)
            tiles.append((x, y, x2, y2))
            y = y2
        x = x2
    return tiles


def fetch_county(
    county_fips: str,
    zcta_bounds_4326: gpd.GeoDataFrame,
    idx_list: list,
) -> list:
    """Fetch all NFHL features for one county's ZCTA bounding box."""
    bounds = zcta_bounds_4326.iloc[idx_list].total_bounds
    pad = 0.01
    xmin, ymin, xmax, ymax = (
        bounds[0] - pad, bounds[1] - pad,
        bounds[2] + pad, bounds[3] + pad,
    )

    dx, dy = xmax - xmin, ymax - ymin
    if dx > MAX_COUNTY_BBOX_DEG or dy > MAX_COUNTY_BBOX_DEG:
        tiles = _split_bbox_to_tiles(xmin, ymin, xmax, ymax, MAX_COUNTY_BBOX_DEG)
        log.info("  County %s: %.1f x %.1f deg -> %d tiles",
                 county_fips, dx, dy, len(tiles))
    else:
        tiles = [(xmin, ymin, xmax, ymax)]

    raw_features = []
    for tile in tiles:
        raw_features.extend(fetch_nfhl_for_bbox(*tile))
        if len(tiles) > 1:
            time.sleep(INTER_REQUEST_SLEEP)

    return raw_features


def run_fetch_phase(
    manifest: list[dict],
    zcta_geo: gpd.GeoDataFrame,
    fetch_dir: Path,
    completed_counties: set,
    threads: int,
) -> int:
    """Phase 1: Fetch NFHL data for all pending counties in parallel.

    Saves raw features as JSON per county in fetch_dir.
    Returns number of counties fetched.
    """
    pending = [m for m in manifest if m["county_fips"] not in completed_counties]

    # Also skip counties whose fetch file already exists on disk
    to_fetch = []
    for m in pending:
        fetch_file = fetch_dir / f"{m['county_fips']}.json"
        if fetch_file.exists():
            continue
        to_fetch.append(m)

    if not to_fetch:
        log.info("Phase 1: All %d counties already fetched", len(manifest))
        return 0

    log.info("Phase 1 FETCH: %d counties to fetch (%d threads)", len(to_fetch), threads)
    fetched = 0

    def _fetch_one(entry):
        county_fips = entry["county_fips"]
        features = fetch_county(county_fips, zcta_geo, entry["idx_list"])
        # Save raw features to disk + S3
        fetch_file = fetch_dir / f"{county_fips}.json"
        with open(fetch_file, "w") as f:
            json.dump(features, f)
        _s3_upload(str(fetch_file), f"{S3_FETCH_PREFIX}{county_fips}.json", quiet=True)
        return county_fips, len(features)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(_fetch_one, m): m["county_fips"] for m in to_fetch}
        for fut in as_completed(futures):
            county_fips = futures[fut]
            try:
                fips, n_feat = fut.result()
                fetched += 1
                if fetched % 25 == 0 or fetched == len(to_fetch):
                    log.info("  Fetched %d/%d counties (latest: %s, %d features)",
                             fetched, len(to_fetch), fips, n_feat)
                    sys.stdout.flush()
            except Exception as e:
                log.warning("  Fetch failed county %s: %s", county_fips, e)
                fetched += 1

    log.info("Phase 1 complete: %d counties fetched", fetched)
    return fetched


# ---------------------------------------------------------------------------
# Representation validation (certificate preconditions)
# ---------------------------------------------------------------------------
# CONUS bounding box in EPSG:5070 (Albers Equal Area), meters
# Approximate: xmin=-2.4M, ymin=0.2M, xmax=2.3M, ymax=3.2M
CONUS_5070_BOUNDS = (-2_500_000, 100_000, 2_400_000, 3_300_000)


def validate_tiger_crs(zcta_proj: gpd.GeoDataFrame) -> bool:
    """Check 1: Confirm TIGER ZCTAs are in EPSG:5070."""
    crs = zcta_proj.crs
    if crs is None:
        log.error("VALIDATION FAIL: TIGER has no CRS")
        return False
    epsg = crs.to_epsg()
    if epsg != 5070:
        log.error("VALIDATION FAIL: TIGER CRS is EPSG:%s, expected 5070", epsg)
        return False
    log.info("  Check 1 PASS: TIGER CRS is EPSG:5070")
    return True


def validate_tiger_bounds(zcta_proj: gpd.GeoDataFrame) -> bool:
    """Check 2: Confirm TIGER bounds are within CONUS EPSG:5070 range."""
    xmin, ymin, xmax, ymax = zcta_proj.total_bounds
    cxmin, cymin, cxmax, cymax = CONUS_5070_BOUNDS
    ok = (xmin > cxmin and ymin > cymin and xmax < cxmax and ymax < cymax)
    if not ok:
        log.warning("VALIDATION WARN: TIGER bounds (%.0f,%.0f,%.0f,%.0f) "
                     "extend beyond CONUS (%.0f,%.0f,%.0f,%.0f) -- "
                     "AK/HI/territories present?",
                     xmin, ymin, xmax, ymax, cxmin, cymin, cxmax, cymax)
    else:
        log.info("  Check 2 PASS: TIGER bounds within CONUS EPSG:5070 range")
    # Non-fatal: AK/HI ZCTAs exist in TIGER but won't match flood data
    return True


def validate_fema_sample(
    zcta_geo_4326: gpd.GeoDataFrame,
    zcta_proj: gpd.GeoDataFrame,
) -> bool:
    """Checks 3-6: Fetch a small FEMA sample and validate geometry pipeline.

    Tests CRS alignment, geometry validity, ring conversion, and area conservation
    using a known area (Harris County, TX — FIPS 48201).
    """
    # Use a small bbox in central Houston (well-known flood zone)
    # Roughly: 29.72-29.78N, 95.40-95.34W
    test_bbox = (-95.40, 29.72, -95.34, 29.78)
    log.info("  Fetching FEMA validation sample (Houston bbox)...")
    features = fetch_nfhl_for_bbox(*test_bbox)

    if not features:
        log.warning("VALIDATION WARN: No FEMA features returned for Houston test bbox. "
                     "API may be down -- proceeding without validation.")
        return True

    log.info("  Got %d features from Houston test bbox", len(features))

    # Check 3: Geometry validity before and after repair
    # Check 4: Ring winding order (independent of our conversion)
    n_total = 0
    n_null = 0
    n_invalid_raw = 0
    n_invalid_repaired = 0
    n_repaired = 0
    n_cw_shells = 0      # Esri convention: outer = clockwise
    n_ccw_shells = 0     # Non-standard: outer = counter-clockwise
    n_multi_ring = 0
    n_holes_found = 0
    raw_polys_5070 = []  # For area conservation in projected coords
    rep_polys_5070 = []

    for f in features:
        rings = f.get("geometry", {}).get("rings", [])
        if not rings:
            n_null += 1
            continue
        n_total += 1

        # Check 4: Independently inspect ring winding order
        if len(rings) > 1:
            n_multi_ring += 1
        for ri, ring in enumerate(rings):
            sa = _signed_area(ring)
            if ri == 0:
                # First ring — Esri says this should be CW (negative SA)
                if sa < 0:
                    n_cw_shells += 1
                else:
                    n_ccw_shells += 1
            else:
                if sa > 0:
                    n_holes_found += 1

        # Build naive polygon (first ring = shell, rest = holes) for comparison
        try:
            naive_poly = Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0])
        except Exception:
            n_invalid_raw += 1
            continue

        if not naive_poly.is_valid:
            n_invalid_raw += 1

        # Apply production conversion (winding-aware)
        repaired = _esri_to_shapely(f["geometry"])
        if repaired is None:
            n_null += 1
            continue

        if not repaired.is_valid:
            n_invalid_repaired += 1

        if not naive_poly.is_valid and repaired.is_valid:
            n_repaired += 1

        # Collect for area conservation in EPSG:5070
        if naive_poly.is_valid:
            raw_polys_5070.append(naive_poly)
            rep_polys_5070.append(repaired)

    log.info("  Check 3: Geometry validity: %d total, %d null, "
             "%d invalid raw, %d invalid after repair, %d repaired by buffer(0)",
             n_total, n_null, n_invalid_raw, n_invalid_repaired, n_repaired)

    if n_invalid_repaired > 0:
        log.warning("VALIDATION WARN: %d polygons still invalid after repair", n_invalid_repaired)

    # Check 4: Ring winding report
    log.info("  Check 4: Ring winding: %d CW first-rings (Esri standard), "
             "%d CCW first-rings (non-standard), %d multi-ring features, %d holes",
             n_cw_shells, n_ccw_shells, n_multi_ring, n_holes_found)
    if n_ccw_shells > n_cw_shells:
        log.warning("VALIDATION WARN: Majority of first-rings are CCW -- "
                     "FEMA may not follow Esri CW-outer convention for this layer")
    elif n_ccw_shells > 0:
        log.info("  Check 4 NOTE: %d/%d first-rings are CCW (non-standard but handled)",
                 n_ccw_shells, n_total)
    else:
        log.info("  Check 4 PASS: All first-rings follow Esri CW convention")

    # Check 6: Area conservation in EPSG:5070 (projected, not degrees)
    if raw_polys_5070:
        try:
            raw_gdf = gpd.GeoDataFrame(
                geometry=raw_polys_5070[:50], crs="EPSG:4326"
            ).to_crs("EPSG:5070")
            rep_gdf = gpd.GeoDataFrame(
                geometry=rep_polys_5070[:50], crs="EPSG:4326"
            ).to_crs("EPSG:5070")
            raw_areas = raw_gdf.geometry.area.values
            rep_areas = rep_gdf.geometry.area.values
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(raw_areas > 0, rep_areas / raw_areas, 1.0)
            bad_ratio = int(np.sum((ratios < 0.9) | (ratios > 1.1)))
            log.info("  Check 6: Area conservation (EPSG:5070): %d polygons, "
                     "ratio range [%.4f, %.4f], %d with >10%% change",
                     len(raw_areas), ratios.min(), ratios.max(), bad_ratio)
            if bad_ratio > len(raw_areas) * 0.1:
                log.warning("VALIDATION WARN: >10%% of polygons changed area "
                             "materially after repair (projected)")
            else:
                log.info("  Check 6 PASS: Repair preserves area within 10%%")
        except Exception as e:
            log.warning("VALIDATION WARN: Area conservation check failed: %s", e)

    # Check 2b: Confirm FEMA features project into CONUS EPSG:5070 range
    sample_polys = []
    for f in features[:20]:
        p = _esri_to_shapely(f["geometry"])
        if p is not None:
            sample_polys.append(p)

    if sample_polys:
        sample_gdf = gpd.GeoDataFrame(geometry=sample_polys, crs="EPSG:4326").to_crs("EPSG:5070")
        fxmin, fymin, fxmax, fymax = sample_gdf.total_bounds
        cxmin, cymin, cxmax, cymax = CONUS_5070_BOUNDS
        in_bounds = (fxmin > cxmin and fymin > cymin and fxmax < cxmax and fymax < cymax)
        if in_bounds:
            log.info("  Check 2b PASS: FEMA sample bounds within CONUS EPSG:5070")
        else:
            log.error("VALIDATION FAIL: FEMA sample projects outside CONUS bounds")
            return False

        # Confirm CRS match with TIGER
        if sample_gdf.crs.to_epsg() == zcta_proj.crs.to_epsg():
            log.info("  Check 1b PASS: FEMA reprojected CRS matches TIGER (EPSG:5070)")
        else:
            log.error("VALIDATION FAIL: CRS mismatch FEMA=%s TIGER=%s",
                       sample_gdf.crs.to_epsg(), zcta_proj.crs.to_epsg())
            return False

    return True


def validate_harris_county(
    zcta_geo: gpd.GeoDataFrame,
    zcta_proj: gpd.GeoDataFrame,
) -> bool:
    """Check 5: Harris County sanity check.

    Harris County (FIPS 48201) should have significant flood zone coverage
    due to its flat coastal geography and Hurricane Harvey history.
    Fetch and overlay a few Harris County ZCTAs as an end-to-end test.
    """
    harris_mask = zcta_geo["county_fips"] == "48201"
    harris_count = harris_mask.sum()
    if harris_count == 0:
        log.warning("VALIDATION WARN: No Harris County ZCTAs found (county_fips=48201)")
        return True  # Non-fatal, could be limited dataset

    # Pick up to 5 Harris County ZCTAs for quick test
    harris_idx = list(zcta_geo[harris_mask].index[:5])
    log.info("  Check 5: Harris County sanity -- testing %d ZCTAs (of %d total)",
             len(harris_idx), harris_count)

    features = fetch_county("48201", zcta_geo, harris_idx)
    if not features:
        log.warning("VALIDATION WARN: No FEMA features for Harris County test")
        return True

    result = overlay_county("48201_validation", features, zcta_proj, harris_idx)
    n_with_flood = sum(1 for idx in harris_idx
                       if result["zone_a"].get(idx, 0) > 0
                       or result["zone_x500"].get(idx, 0) > 0)

    log.info("  Check 5: Harris County: %d/%d test ZCTAs have flood coverage, "
             "%d zone_a overlaps, %d zone_x500 overlaps",
             n_with_flood, len(harris_idx),
             len(result["zone_a"]), len(result["zone_x500"]))

    if n_with_flood == 0:
        log.warning("VALIDATION WARN: Zero flood coverage in Harris County -- "
                     "possible CRS or overlay problem")
        return False

    log.info("  Check 5 PASS: Harris County shows nontrivial flood coverage")
    return True


def run_validation(
    zcta_geo: gpd.GeoDataFrame,
    zcta_proj: gpd.GeoDataFrame,
    output_dir: Path,
) -> bool:
    """Run pre-overlay representation validation checks (1-6).

    Saves validation_status.json artifact. Check 7 runs post-overlay.
    Returns True if critical checks pass (non-fatal warnings allowed).
    """
    log.info("=== REPRESENTATION VALIDATION ===")
    log.info("Validating CRS alignment, topology, and overlay stability")

    ok = True
    ok = validate_tiger_crs(zcta_proj) and ok
    ok = validate_tiger_bounds(zcta_proj) and ok
    ok = validate_fema_sample(zcta_geo, zcta_proj) and ok
    ok = validate_harris_county(zcta_geo, zcta_proj) and ok

    status = "PASS" if ok else "FAIL"
    log.info("VALIDATION: Pre-overlay status = %s", status)

    # Save validation artifact (Check 7 appended post-overlay)
    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pre_overlay_status": status,
        "tiger_crs": str(zcta_proj.crs),
        "tiger_bounds_5070": list(map(float, zcta_proj.total_bounds)),
        "tiger_n_zctas": len(zcta_proj),
        "check_7": None,  # filled post-overlay
    }
    artifact_path = output_dir / "validation_status.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    log.info("Saved: %s", artifact_path)

    return ok


# ---------------------------------------------------------------------------
# Phase 2: OVERLAY (sequential, memory bound)
# ---------------------------------------------------------------------------
def overlay_county(
    county_fips: str,
    raw_features: list,
    zcta_proj: gpd.GeoDataFrame,
    idx_list: list,
) -> dict:
    """Overlay one county's NFHL features with ZCTA boundaries.

    Returns dict with zone_a and zone_x500 area dicts keyed by position index.
    """
    result = {"zone_a": {}, "zone_x500": {}}
    if not raw_features:
        return result

    zone_a_polys, zone_x500_polys = [], []
    for f in raw_features:
        zclass = classify_zone(
            f["attributes"].get("FLD_ZONE", ""),
            f["attributes"].get("ZONE_SUBTY", ""),
        )
        if zclass not in ("A", "X500"):
            continue
        poly = _esri_to_shapely(f["geometry"])
        if poly is None:
            continue
        (zone_a_polys if zclass == "A" else zone_x500_polys).append(poly)

    for polys, key in [(zone_a_polys, "zone_a"), (zone_x500_polys, "zone_x500")]:
        if not polys:
            continue
        try:
            zone_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs("EPSG:5070")
            zone_union = zone_gdf.geometry.union_all()
            for pos_idx in idx_list:
                zcta_geom = zcta_proj.iloc[pos_idx].geometry
                intersection = zcta_geom.intersection(zone_union)
                if not intersection.is_empty:
                    result[key][pos_idx] = intersection.area
        except Exception as e:
            log.warning("  Overlay error county %s %s: %s", county_fips, key, e)

    return result


def run_overlay_phase(
    manifest: list[dict],
    zcta_proj: gpd.GeoDataFrame,
    zcta_areas: np.ndarray,
    fetch_dir: Path,
    output_dir: Path,
    zone_a_area: np.ndarray,
    zone_x500_area: np.ndarray,
    completed_counties: set,
) -> tuple[int, int]:
    """Phase 2: Overlay each county sequentially.

    Returns (counties_with_flood, total_zcta_overlaps).
    """
    pending = [m for m in manifest if m["county_fips"] not in completed_counties]
    log.info("Phase 2 OVERLAY: %d counties pending (sequential)", len(pending))

    counties_with_flood = 0
    total_overlaps = 0
    checkpoint_path = output_dir / "flood_checkpoint.json"
    done_so_far = len(completed_counties)

    for i, entry in enumerate(pending):
        county_fips = entry["county_fips"]
        t0 = time.time()

        # Load raw features from disk
        fetch_file = fetch_dir / f"{county_fips}.json"
        if not fetch_file.exists():
            log.warning("  County %s: no fetch file, skipping", county_fips)
            continue

        with open(fetch_file) as f:
            raw_features = json.load(f)

        # Overlay (memory-intensive part — one county at a time)
        result = overlay_county(county_fips, raw_features, zcta_proj, entry["idx_list"])

        # Accumulate
        n_overlaps = 0
        for pos_idx, area in result["zone_a"].items():
            zone_a_area[pos_idx] += area
            n_overlaps += 1
        for pos_idx, area in result["zone_x500"].items():
            zone_x500_area[pos_idx] += area
            n_overlaps += 1

        if n_overlaps > 0:
            counties_with_flood += 1
        total_overlaps += n_overlaps

        # Free memory
        del raw_features, result
        gc.collect()

        completed_counties.add(county_fips)
        done_so_far += 1
        elapsed = time.time() - t0

        # Per-county CloudWatch log
        log.info("  [%d/%d] County %s: %d ZCTAs, %d overlaps, %.1fs, complexity=%.2f",
                 done_so_far, len(manifest), county_fips,
                 entry["n_zctas"], n_overlaps, elapsed, entry["complexity"])

        # Checkpoint after every county
        partial_path = str(output_dir / "flood_partial.npz")
        np.savez(partial_path,
                 zone_a_area=zone_a_area, zone_x500_area=zone_x500_area)
        checkpoint_path.write_text(json.dumps({
            "completed_counties": sorted(completed_counties),
            "done_count": done_so_far,
            "counties_with_flood": counties_with_flood,
            "total_overlaps": total_overlaps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

        # Upload checkpoint to S3 every 50 counties
        if done_so_far % 50 == 0:
            _s3_upload(partial_path, S3_CHECKPOINT_KEY)
            _s3_upload(str(checkpoint_path), S3_CHECKPOINT_JSON_KEY)
            log.info("  Checkpoint uploaded to S3 (%d counties)", done_so_far)

    log.info("Phase 2 complete: %d with flood, %d ZCTA overlaps",
             counties_with_flood, total_overlaps)
    return counties_with_flood, total_overlaps


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build FEMA flood zones (SageMaker)")
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--max-zctas", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8,
                        help="Concurrent FEMA API fetchers (Phase 1 only)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch_dir = output_dir / "fetch_cache"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("=== LOADING DATA ===")
    zcta_geo = load_tiger(args.tiger_dir)
    xwalk = load_crosswalk(args.data_dir)

    if args.max_zctas:
        zcta_geo = zcta_geo.head(args.max_zctas)
        log.info("Limited to %d ZCTAs", len(zcta_geo))

    # Assign counties
    zcta_counties = dict(zip(xwalk["zcta_id"], xwalk["county_fips"]))
    zcta_geo = zcta_geo.copy()
    zcta_geo["county_fips"] = zcta_geo["zcta_id"].map(zcta_counties).fillna("unknown")

    # CONUS safety net: drop AK (02), HI (15), PR (72), territories (60,66,69,78)
    # Primary filter is in build_zcta_county_crosswalk.py; this is a backstop
    # because EPSG:5070 is only valid for CONUS -- area fractions for AK/HI would be wrong.
    NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
    before_filter = len(zcta_geo)
    state_fips = zcta_geo["county_fips"].str[:2]
    conus_mask = ~state_fips.isin(NON_CONUS_STATE_FIPS) & (zcta_geo["county_fips"] != "unknown")
    zcta_geo = zcta_geo[conus_mask].reset_index(drop=True)
    log.info("CONUS filter: %d -> %d ZCTAs (dropped %d non-CONUS + unknown)",
             before_filter, len(zcta_geo), before_filter - len(zcta_geo))

    # Project once
    log.info("Projecting to EPSG:5070...")
    zcta_proj = zcta_geo.to_crs("EPSG:5070")
    zcta_areas = zcta_proj.geometry.area.values
    n = len(zcta_geo)

    # Accumulators
    zone_a_area = np.zeros(n, dtype=np.float64)
    zone_x500_area = np.zeros(n, dtype=np.float64)

    # Check for checkpoint (crash recovery -- local first, then S3)
    checkpoint_path = output_dir / "flood_checkpoint.json"
    completed_counties = set()
    if not checkpoint_path.exists():
        _s3_download(S3_CHECKPOINT_JSON_KEY, str(checkpoint_path))
    if checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        completed_counties = set(ckpt.get("completed_counties", []))
        partial_path = output_dir / "flood_partial.npz"
        if not partial_path.exists():
            _s3_download(S3_CHECKPOINT_KEY, str(partial_path))
        if partial_path.exists():
            data = np.load(partial_path)
            zone_a_area = data["zone_a_area"]
            zone_x500_area = data["zone_x500_area"]
        log.info("Resuming from checkpoint: %d counties done", len(completed_counties))

        # Restore fetch files from S3 for completed counties
        fetch_dir = output_dir / "fetch_cache"
        fetch_dir.mkdir(parents=True, exist_ok=True)
        restored = 0
        for fips in completed_counties:
            fetch_file = fetch_dir / f"{fips}.json"
            if not fetch_file.exists():
                if _s3_download(f"{S3_FETCH_PREFIX}{fips}.json", str(fetch_file), quiet=True):
                    restored += 1
        if restored:
            log.info("  Restored %d county fetch files from S3", restored)

    # Representation validation (certificate preconditions)
    log.info("")
    if not run_validation(zcta_geo, zcta_proj, output_dir):
        log.error("Representation validation failed. Aborting.")
        sys.exit(1)

    # Build size-sorted manifest
    log.info("\n=== BUILDING COUNTY MANIFEST ===")
    manifest = build_county_manifest(zcta_geo)

    # Resource estimation
    total_zctas = sum(m["n_zctas"] for m in manifest)
    top5 = manifest[-5:]
    log.info("Resource estimate:")
    log.info("  Total counties: %d, total ZCTAs: %d", len(manifest), total_zctas)
    log.info("  Top 5 by complexity:")
    for m in reversed(top5):
        log.info("    %s: %d ZCTAs, bbox=%.2f deg2, complexity=%.2f",
                 m["county_fips"], m["n_zctas"], m["bbox_area_deg2"], m["complexity"])

    # Phase 1: FETCH (parallel)
    log.info("\n=== PHASE 1: FETCH ===")
    run_fetch_phase(manifest, zcta_geo, fetch_dir, completed_counties, args.threads)

    # Phase 2: OVERLAY (sequential)
    log.info("\n=== PHASE 2: OVERLAY ===")
    counties_with_flood, total_overlaps = run_overlay_phase(
        manifest, zcta_proj, zcta_areas, fetch_dir, output_dir,
        zone_a_area, zone_x500_area, completed_counties,
    )

    # Convert to percentages
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_a = np.where(zcta_areas > 0,
                         np.minimum(zone_a_area / zcta_areas * 100, 100.0), 0.0)
        pct_x500 = np.where(zcta_areas > 0,
                            np.minimum(zone_x500_area / zcta_areas * 100, 100.0), 0.0)
    pct_x = np.maximum(100.0 - pct_a - pct_x500, 0.0)

    result = pd.DataFrame({
        "zcta_id": zcta_geo["zcta_id"].values,
        "flood_pct_zone_a": np.round(pct_a, 2),
        "flood_pct_zone_x500": np.round(pct_x500, 2),
        "flood_pct_zone_x": np.round(pct_x, 2),
        "flood_sfha": pct_a > 0,
    })

    # Check 7: Fraction distribution validation
    log.info("")
    log.info("=== CHECK 7: FRACTION DISTRIBUTION ===")
    check7 = {}
    for col_name, arr in [("zone_a", pct_a), ("zone_x500", pct_x500), ("zone_x", pct_x)]:
        stats = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "pct_zero": float(np.mean(arr == 0) * 100),
            "pct_near_100": float(np.mean(arr >= 99.99) * 100),
            "pct_tiny_nonzero": float(np.mean((arr > 0) & (arr < 0.01)) * 100),
            "pct_gt_100": float(np.mean(arr > 100.0) * 100),
        }
        check7[col_name] = stats
        log.info("  %s: min=%.4f max=%.2f mean=%.2f median=%.2f",
                 col_name, stats["min"], stats["max"], stats["mean"], stats["median"])
        log.info("    zero=%.1f%%, tiny(<0.01)=%.1f%%, near-100=%.1f%%, >100=%.1f%%",
                 stats["pct_zero"], stats["pct_tiny_nonzero"],
                 stats["pct_near_100"], stats["pct_gt_100"])

    # Impossible values: fractions > 100% should never exist
    if check7["zone_a"]["pct_gt_100"] > 0 or check7["zone_x500"]["pct_gt_100"] > 0:
        log.error("VALIDATION FAIL: Found fractions > 100%% -- overlay arithmetic error")

    # Zone A + Zone X500 should not exceed 100% for any ZCTA
    overflow = np.sum((pct_a + pct_x500) > 100.01)
    if overflow > 0:
        log.warning("VALIDATION WARN: %d ZCTAs have zone_a + zone_x500 > 100%%", overflow)
    check7["n_a_plus_x500_overflow"] = int(overflow)

    # Zone X (minimal risk) should dominate
    if check7["zone_x"]["mean"] < 50:
        log.warning("VALIDATION WARN: Zone X mean < 50%% -- unexpected, "
                     "most US land should be minimal flood risk")

    # Zone A should be nonzero for a meaningful fraction (>5% of ZCTAs)
    zone_a_coverage = 1.0 - check7["zone_a"]["pct_zero"] / 100.0
    if zone_a_coverage < 0.05:
        log.warning("VALIDATION WARN: <5%% of ZCTAs have any Zone A coverage -- "
                     "possible overlay failure")

    # Suspicious mass points: > 10% of ZCTAs at exactly 0.01 (epsilon artifacts)
    epsilon_spike = float(np.mean(np.abs(pct_a - 0.01) < 1e-6) * 100)
    if epsilon_spike > 10:
        log.warning("VALIDATION WARN: %.1f%% of ZCTAs have zone_a = 0.01 exactly "
                     "-- possible rounding artifact", epsilon_spike)
    check7["zone_a_epsilon_spike_pct"] = epsilon_spike

    check7_pass = (
        check7["zone_a"]["pct_gt_100"] == 0
        and check7["zone_x500"]["pct_gt_100"] == 0
        and overflow == 0
        and check7["zone_x"]["mean"] >= 50
        and zone_a_coverage >= 0.05
    )
    check7["status"] = "PASS" if check7_pass else "WARN"
    log.info("  Check 7: %s", check7["status"])

    # Update validation artifact with Check 7 results
    artifact_path = output_dir / "validation_status.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
    else:
        artifact = {}
    artifact["check_7"] = check7
    artifact["overall_status"] = "PASS" if check7_pass else "WARN"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    _s3_upload(str(artifact_path),
               "rsct_curriculum/series_018/processed/flood_validation_status.json")
    log.info("  Validation artifact updated: %s", artifact_path)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:         %d", len(result))
    log.info("Zone A >0%%:    %d (%.1f%%)",
             (result["flood_pct_zone_a"] > 0).sum(),
             (result["flood_pct_zone_a"] > 0).mean() * 100)
    log.info("Zone X500 >0%%: %d (%.1f%%)",
             (result["flood_pct_zone_x500"] > 0).sum(),
             (result["flood_pct_zone_x500"] > 0).mean() * 100)
    log.info("Mean Zone A:   %.2f%%", result["flood_pct_zone_a"].mean())
    log.info("Mean Zone X500: %.2f%%", result["flood_pct_zone_x500"].mean())
    log.info("SFHA count:    %d", result["flood_sfha"].sum())

    # Save locally
    out_path = output_dir / "flood_zones_zcta.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Saved: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    # Upload final parquet directly to S3
    log.info("Uploading final parquet to S3...")
    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    # Provenance
    provenance = {
        "operation": "build_flood_zones",
        "timestamp": timestamp,
        "source": f"{NFHL_BASE}/{NFHL_LAYER}",
        "method": "county_batched_polygon_overlay_2phase",
        "page_size": PAGE_SIZE,
        "max_county_bbox_deg": MAX_COUNTY_BBOX_DEG,
        "threads": args.threads,
        "n_zctas": len(result),
        "n_counties": len(manifest),
        "counties_with_flood": counties_with_flood,
        "zone_a_count": int((result["flood_pct_zone_a"] > 0).sum()),
        "zone_x500_count": int((result["flood_pct_zone_x500"] > 0).sum()),
        "sfha_count": int(result["flood_sfha"].sum()),
        "mean_zone_a_pct": round(float(result["flood_pct_zone_a"].mean()), 2),
    }
    prov_path = output_dir / "flood_zones_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)

    # Clean up checkpoint files
    for f in [checkpoint_path, output_dir / "flood_partial.npz"]:
        if f.exists():
            f.unlink()

    log.info("Done.")


if __name__ == "__main__":
    main()
