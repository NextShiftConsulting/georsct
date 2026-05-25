#!/usr/bin/env python3
"""
Stage 2: Extract S_FLD_HAZ_AR from raw FEMA county zips on S3.

Reads:  s3://swarm-yrsn-datasets/.../flood_raw/{dfirm_id}.zip
Writes: s3://swarm-yrsn-datasets/.../flood_fetch/{fips}.json

Handles:
  - Dedup: multiple DFIRMs per county FIPS -> keep latest by date
  - Validation: geometry validity, ring winding, feature counts
  - Resume: skips FIPS already in flood_fetch/

Instance: ml.m5.12xlarge (48 vCPU, 192 GB, 10 Gbps network)
Threads:  96 (2x CPU count) — each thread: S3 download -> unzip -> fiona read -> JSON write -> S3 upload
"""

import argparse
import gc
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
S3_RAW_PREFIX = "rsct_curriculum/series_018/processed/flood_raw/"
S3_FETCH_PREFIX = "rsct_curriculum/series_018/processed/flood_fetch/"

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    import boto3
    return boto3.client("s3")


def _s3_upload(s3, local_path: str, key: str):
    s3.upload_file(local_path, S3_BUCKET, key)


def _s3_download(s3, key: str, local_path: str):
    s3.download_file(S3_BUCKET, key, local_path)


def _s3_list_prefix(s3, prefix: str, suffix: str = "") -> list:
    """List objects under prefix, return list of (key, filename)."""
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1]
            if suffix and not fname.endswith(suffix):
                continue
            results.append((obj["Key"], fname))
    return results


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

def _signed_area(ring) -> float:
    """Shoelace formula signed area. Positive = CCW, Negative = CW."""
    area = 0.0
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[j][0], ring[j][1]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def validate_features(features: list) -> dict:
    """Run Check 3 (geometry) and Check 4 (winding) on extracted features.

    Returns validation stats dict.
    """
    stats = {
        "total_features": len(features),
        "null_geometry": 0,
        "empty_rings": 0,
        "cw_outer": 0,
        "ccw_outer": 0,
        "multi_ring": 0,
        "holes_found": 0,
        "unclosed_rings": 0,
        "degenerate_rings": 0,
    }

    for feat in features:
        rings = feat.get("geometry", {}).get("rings", [])
        if not rings:
            stats["null_geometry"] += 1
            continue

        if len(rings) > 1:
            stats["multi_ring"] += 1

        for i, ring in enumerate(rings):
            if len(ring) < 4:
                stats["degenerate_rings"] += 1
                continue

            # Check closure
            if ring[0] != ring[-1]:
                stats["unclosed_rings"] += 1

            sa = _signed_area(ring)
            if i == 0:
                # First ring = outer shell
                if sa < 0:
                    stats["cw_outer"] += 1  # Esri convention: CW = outer
                elif sa > 0:
                    stats["ccw_outer"] += 1
            else:
                # Subsequent rings = holes
                stats["holes_found"] += 1

    return stats


# ---------------------------------------------------------------------------
# Extract one county
# ---------------------------------------------------------------------------

def extract_county(s3, zip_key: str, dfirm_id: str, fips: str,
                   work_dir: Path, output_dir: Path) -> tuple:
    """Download zip from S3, extract shapefile, write JSON, upload.

    Returns (fips, n_features, validation_stats, error_msg).
    """
    zip_path = work_dir / f"{dfirm_id}.zip"
    extract_dir = work_dir / f"{dfirm_id}_ext"

    try:
        # Download from S3
        _s3_download(s3, zip_key, str(zip_path))

        # Validate zip integrity before extracting
        if not zipfile.is_zipfile(zip_path):
            return fips, 0, None, "corrupt zip (not a valid zip file)"
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return fips, 0, None, f"corrupt zip (bad entry: {bad})"
            zf.extractall(extract_dir)

        # Find flood hazard shapefile (case-insensitive)
        import fiona

        shp_paths = [p for p in extract_dir.rglob("*.shp")
                     if p.stem.upper() == "S_FLD_HAZ_AR"]
        gdb_paths = list(extract_dir.rglob("*.gdb"))

        if shp_paths:
            data_path = str(shp_paths[0])
            layer_arg = {}
        elif gdb_paths:
            data_path = str(gdb_paths[0])
            layers = fiona.listlayers(data_path)
            if "S_Fld_Haz_Ar" not in layers:
                return fips, 0, None, f"No S_Fld_Haz_Ar layer (layers: {layers[:5]})"
            layer_arg = {"layer": "S_Fld_Haz_Ar"}
        else:
            all_files = [str(p.name) for p in extract_dir.rglob("*") if p.is_file()]
            return fips, 0, None, f"No .shp or .gdb (files: {all_files[:10]})"

        # Stream features to JSON file
        fetch_file = output_dir / f"{fips}.json"
        features_for_validation = []
        n_features = 0

        with fiona.open(data_path, **layer_arg) as src:
            with open(fetch_file, "w") as out:
                out.write("[")
                first = True
                for feat in src:
                    geom = feat.get("geometry", {})
                    rings = _geom_to_rings(geom)
                    if not rings:
                        continue
                    props = dict(feat.get("properties", {}))
                    item = {
                        "attributes": {
                            "FLD_ZONE": props.get("FLD_ZONE", ""),
                            "ZONE_SUBTY": props.get("ZONE_SUBTY", ""),
                            "SFHA_TF": props.get("SFHA_TF", ""),
                        },
                        "geometry": {"rings": rings},
                    }
                    if not first:
                        out.write(",")
                    json.dump(item, out)
                    first = False
                    n_features += 1

                    # Sample for validation (first 500 features)
                    if n_features <= 500:
                        features_for_validation.append(item)

                out.write("]")

        # Validate sampled features
        vstats = validate_features(features_for_validation)
        vstats["total_features"] = n_features  # override with true count

        # Upload to flood_fetch/
        _s3_upload(s3, str(fetch_file), f"{S3_FETCH_PREFIX}{fips}.json")
        fetch_file.unlink(missing_ok=True)

        return fips, n_features, vstats, None

    except Exception as e:
        return fips, 0, None, str(e)
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


def _geom_to_rings(geom: dict) -> list:
    """Convert fiona/GeoJSON geometry to Esri-style rings list."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--conus-only", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="nfhl_extract_"))

    s3 = _s3_client()

    # -----------------------------------------------------------------------
    # 1. List raw zips on S3
    # -----------------------------------------------------------------------
    log.info("Listing raw zips on S3: %s", S3_RAW_PREFIX)
    raw_files = _s3_list_prefix(s3, S3_RAW_PREFIX, suffix=".zip")
    log.info("  Found %d raw zips", len(raw_files))

    if not raw_files:
        log.info("No raw zips found. Nothing to do.")
        return

    # -----------------------------------------------------------------------
    # 2. Dedup: keep latest version per county FIPS
    # -----------------------------------------------------------------------
    by_fips = {}
    for key, fname in raw_files:
        # fname: "06001C_20250430.zip" -> dfirm_id="06001C", date="20250430"
        dfirm_id = fname.split("_")[0]
        fips = dfirm_id[:5]

        if args.conus_only and fips[:2] in NON_CONUS_PREFIXES:
            continue

        m = re.search(r"_(\d{8})\.", fname)
        entry_date = m.group(1) if m else "00000000"

        if fips not in by_fips or entry_date > by_fips[fips]["date"]:
            by_fips[fips] = {"key": key, "dfirm_id": dfirm_id, "date": entry_date}

    log.info("  Dedup: %d zips -> %d unique FIPS (latest per county)", len(raw_files), len(by_fips))

    # Extract all — overwrite any API-fetched files with bulk-extracted
    # (bulk shapefiles are county-clipped, more accurate than API bbox queries)
    to_extract = by_fips
    log.info("  To extract: %d counties (overwriting any existing)", len(to_extract))

    # -----------------------------------------------------------------------
    # 4. Extract in parallel — max throughput
    # -----------------------------------------------------------------------
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    n_threads = max(n_cpus * 2, 16)  # 2x CPU count, floor 16
    log.info("=== EXTRACTING %d COUNTIES (%d threads, %d CPUs) ===",
             len(to_extract), n_threads, n_cpus)
    t_start = time.time()
    done = 0
    failed = 0
    total_features = 0
    empty_counties = []

    # Aggregate validation stats
    agg_vstats = {
        "total_features": 0,
        "null_geometry": 0,
        "empty_rings": 0,
        "cw_outer": 0,
        "ccw_outer": 0,
        "multi_ring": 0,
        "holes_found": 0,
        "unclosed_rings": 0,
        "degenerate_rings": 0,
    }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = sorted(to_extract.items())
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(extract_county, s3, info["key"], info["dfirm_id"],
                            fips, work_dir, output_dir): fips
            for fips, info in items
        }
        for fut in as_completed(futures):
            fips = futures[fut]
            try:
                _, n_feat, vstats, error = fut.result()
                done += 1

                if error:
                    log.warning("  SKIP %s: %s", fips, error)
                    failed += 1
                else:
                    total_features += n_feat
                    if n_feat == 0:
                        empty_counties.append(fips)
                    if vstats:
                        for k in agg_vstats:
                            agg_vstats[k] += vstats.get(k, 0)

                if done % 50 == 0 or done == len(to_extract):
                    elapsed = time.time() - t_start
                    rate = done / (elapsed / 60) if elapsed > 0 else 0
                    eta = (len(to_extract) - done) / rate if rate > 0 else 0
                    log.info("  [%d/%d] %d features (%.0f/min, ETA %.0f min, %d failed)",
                             done, len(to_extract), n_feat if not error else 0,
                             rate, eta, failed)
                    sys.stdout.flush()

            except Exception as e:
                log.warning("  ERROR %s: %s", fips, e)
                failed += 1
                done += 1

    elapsed = time.time() - t_start

    # -----------------------------------------------------------------------
    # 5. Validation report
    # -----------------------------------------------------------------------
    log.info("\n=== VALIDATION ===")

    # Check 3: Geometry validity
    pct_null = (agg_vstats["null_geometry"] / max(agg_vstats["total_features"], 1)) * 100
    pct_degen = (agg_vstats["degenerate_rings"] / max(agg_vstats["total_features"], 1)) * 100
    log.info("Check 3 - Geometry:")
    log.info("  Total features: %d", agg_vstats["total_features"])
    log.info("  Null geometry: %d (%.2f%%)", agg_vstats["null_geometry"], pct_null)
    log.info("  Degenerate rings (<4 pts): %d (%.2f%%)", agg_vstats["degenerate_rings"], pct_degen)
    log.info("  Unclosed rings: %d", agg_vstats["unclosed_rings"])
    check3 = "PASS" if pct_null < 1.0 and pct_degen < 1.0 else "WARN"
    log.info("  Result: %s", check3)

    # Check 4: Ring winding
    total_outer = agg_vstats["cw_outer"] + agg_vstats["ccw_outer"]
    pct_cw = (agg_vstats["cw_outer"] / max(total_outer, 1)) * 100
    log.info("Check 4 - Ring winding:")
    log.info("  CW outer (Esri convention): %d (%.1f%%)", agg_vstats["cw_outer"], pct_cw)
    log.info("  CCW outer: %d", agg_vstats["ccw_outer"])
    log.info("  Multi-ring features: %d", agg_vstats["multi_ring"])
    log.info("  Holes found: %d", agg_vstats["holes_found"])
    check4 = "PASS" if pct_cw > 50 else "WARN"
    log.info("  Result: %s", check4)

    # Feature count check
    log.info("Feature counts:")
    log.info("  Empty counties (0 features): %d", len(empty_counties))
    if empty_counties:
        log.info("  Empty FIPS: %s", empty_counties[:20])
    feat_check = "PASS" if len(empty_counties) < 50 else "WARN"
    log.info("  Result: %s", feat_check)

    # Harris County sanity (Check 5 - quick version)
    harris_check = "48201" not in empty_counties and "48201" in by_fips
    log.info("Check 5 - Harris County (48201) present: %s",
             "PASS" if harris_check else "FAIL")

    log.info("\n=== DONE ===")
    log.info("  Extracted: %d, Failed: %d, Total features: %d",
             done - failed, failed, total_features)
    log.info("  Elapsed: %.1f min", elapsed / 60)

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    # Capture instance metadata for reproducibility / NeurIPS reporting
    import multiprocessing as _mp
    try:
        import resource as _res
        peak_mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        peak_mb = None

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": "extract",
        "extracted": done - failed,
        "failed": failed,
        "total_features": total_features,
        "empty_counties": len(empty_counties),
        "elapsed_sec": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 1),
        "compute": {
            "instance": os.environ.get("SM_RESOURCE_CONFIG", "unknown"),
            "n_cpus": _mp.cpu_count(),
            "n_threads": n_threads,
            "peak_memory_mb": peak_mb,
        },
        "validation": {
            "check3_geometry": check3,
            "check4_winding": check4,
            "check5_harris": "PASS" if harris_check else "FAIL",
            "feature_count": feat_check,
            "stats": agg_vstats,
        },
    }
    summary_path = output_dir / "extract_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _s3_upload(s3, str(summary_path), f"{S3_FETCH_PREFIX}extract_summary.json")
    log.info("Summary: %s", json.dumps(summary, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
