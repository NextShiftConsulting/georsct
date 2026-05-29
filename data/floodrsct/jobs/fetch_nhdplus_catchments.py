#!/usr/bin/env python3
"""
fetch_nhdplus_catchments.py -- SageMaker job: download NHDPlus V2 catchment
polygons and flowline attributes for Riverside-Coachella and Houston.

THIS SCRIPT ONLY FETCHES AND NORMALIZES RAW DATA.
ZCTA catchment joins (build_catchment_features) happen in build_event_dataset.py.

Source: NHDPlus V2 national seamless dataset
  EPA FTP: https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/
  Catchments shapefile: NHDPlusV21_NationalData_Catchment_Shape_01.7z (national)
  Or per-VPU (Vector Processing Unit) downloads which are smaller.

Target VPUs (Vector Processing Units):
  18  — California (covers Riverside-Coachella / Salton Sea / Mojave)
  12  — Texas Gulf (covers Houston / Harris County)

Output:
  s3://swarm-floodrsct-data/raw/nhdplus/catchments/v2/catchments_{vpu}.parquet
  s3://swarm-floodrsct-data/manifests/nhdplus_v2_catchments/v2/manifest.json

Schema: NHDPlusID (COMID), AreaSqKm, geometry (WKT), VPU, HUC4
"""

import hashlib
import io
import logging
import sys
import zipfile
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DST_BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/nhdplus/catchments/v2"
VERSION = "v2"
DATASET = "nhdplus_v2_catchments"

# NHDPlus V2 VPU download URLs (EPA S3 open data)
# Each VPU zip contains NHDPlusCatchment shapefile + NHDFlowline + attributes
VPU_DOWNLOADS = {
    "18": {
        "name": "California",
        "url": (
            "https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/"
            "Data/NHDPlusCA/NHDPlusV21_CA_18_NHDPlusCatchment_01.7z"
        ),
        # Fallback to NHDPlus HR API or alternative
        "fallback_url": (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/"
            "NHD/HU4/HighResolution/Shape/NHD_H_1807_HU4_Shape.zip"
        ),
        "bbox": {"minx": -118.5, "miny": 33.0, "maxx": -114.0, "maxy": 35.5},
    },
    "12": {
        "name": "Texas Gulf",
        "url": (
            "https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/"
            "Data/NHDPlusTX/NHDPlusV21_TX_12_NHDPlusCatchment_01.7z"
        ),
        "fallback_url": (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/"
            "NHD/HU4/HighResolution/Shape/NHD_H_1204_HU4_Shape.zip"
        ),
        "bbox": {"minx": -96.0, "miny": 29.0, "maxx": -94.5, "maxy": 30.5},
    },
}


def try_download(url: str, timeout: int = 600) -> bytes:
    resp = requests.get(url, timeout=timeout, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    data = resp.content
    log.info("Downloaded %.1f MB from %s", len(data) / 1e6, url)
    return data


def extract_catchments_from_zip(zip_bytes: bytes, bbox: dict) -> "gpd.GeoDataFrame":
    """Extract catchment shapefile from ZIP, filter to bbox, return GeoDataFrame."""
    try:
        import geopandas as gpd
    except ImportError:
        raise

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile:
            # 7z files need py7zr
            try:
                import py7zr
                with py7zr.SevenZipFile(io.BytesIO(zip_bytes)) as z:
                    z.extractall(tmpdir)
            except Exception as exc:
                raise RuntimeError(f"Cannot extract archive: {exc}")
        else:
            zf.extractall(tmpdir)

        shp_files = list(Path(tmpdir).rglob("*Catchment*.shp"))
        if not shp_files:
            shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise RuntimeError("No shapefile found in archive")

        shp_path = shp_files[0]
        log.info("Reading: %s", shp_path.name)
        gdf = gpd.read_file(shp_path)

    gdf = gdf.to_crs("EPSG:4326")
    # Filter to bounding box
    gdf = gdf.cx[bbox["minx"]:bbox["maxx"], bbox["miny"]:bbox["maxy"]].copy()
    log.info("Catchments in bbox: %d", len(gdf))

    # Normalize key columns
    col_map = {}
    for col in gdf.columns:
        lc = col.lower()
        if lc in ("featureid", "comid", "nhdplusid"):
            col_map[col] = "comid"
        elif "areasqkm" in lc or "area_sq" in lc:
            col_map[col] = "area_sq_km"
    gdf = gdf.rename(columns=col_map)

    keep_cols = [c for c in ["comid", "area_sq_km", "geometry"] if c in gdf.columns]
    return gdf[keep_cols].copy()


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def main() -> None:
    try:
        import geopandas as gpd
    except ImportError:
        log.error("geopandas required: pip install geopandas")
        sys.exit(1)

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    total_uploaded = 0

    for vpu, spec in VPU_DOWNLOADS.items():
        dst_key = f"{DST_PREFIX}/catchments_vpu{vpu}.parquet"
        if s3_key_exists(s3, DST_BUCKET, dst_key):
            log.info("Already exists: s3://%s/%s — skipping", DST_BUCKET, dst_key)
            total_uploaded += 1
            continue

        log.info("VPU %s (%s): downloading...", vpu, spec["name"])
        zip_bytes = None
        for url in [spec["url"], spec.get("fallback_url")]:
            if not url:
                continue
            try:
                zip_bytes = try_download(url)
                break
            except Exception as exc:
                log.warning("URL failed: %s — %s", url, exc)

        if zip_bytes is None:
            log.error("All download URLs failed for VPU %s — skipping", vpu)
            continue

        checksum = hashlib.sha256(zip_bytes).hexdigest()
        try:
            gdf = extract_catchments_from_zip(zip_bytes, spec["bbox"])
        except Exception as exc:
            log.error("Extract failed for VPU %s: %s", vpu, exc)
            continue

        local = f"/tmp/catchments_vpu{vpu}.parquet"
        gdf.to_parquet(local, index=False)
        s3.upload_file(local, DST_BUCKET, dst_key)
        log.info("Uploaded s3://%s/%s (%d catchments)", DST_BUCKET, dst_key, len(gdf))
        total_uploaded += 1
        sys.stdout.flush()

    write_manifest(
        s3=s3,
        dataset=DATASET,
        version=VERSION,
        source_url="https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/",
        s3_key=DST_PREFIX + "/",
        crs="EPSG:4326",
        record_count=total_uploaded,
        license_="Public Domain — USGS/EPA NHDPlus",
        notes=(
            "NHDPlus V2 catchment polygons for VPU 18 (CA) and VPU 12 (TX Gulf). "
            "ZCTA catchment joins (upstream_catchment_km2, wash_segment_id, bayou_segment_id) "
            "computed in build_event_dataset.py via build_catchment_features()."
        ),
    )

    log.info("fetch_nhdplus_catchments complete: %d VPUs uploaded", total_uploaded)


if __name__ == "__main__":
    main()
