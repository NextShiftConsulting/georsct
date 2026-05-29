#!/usr/bin/env python3
"""
fetch_mtbs_burn_scars.py -- SageMaker job: download USGS MTBS burn perimeter
shapefiles (2015-2023) and upload raw GeoParquet to S3.

THIS SCRIPT ONLY FETCHES AND NORMALIZES RAW DATA.
ZCTA burn-scar overlay (build_burn_scar_features) happens in build_event_dataset.py.

Source: USGS Monitoring Trends in Burn Severity (MTBS)
  Direct download: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Data/data/
  Perimeter ZIP: mtbs_perims_DD.zip (national, all years)

Output:
  s3://swarm-floodrsct-data/raw/mtbs/perimeters/v2023/burn_perims.parquet (GeoParquet)
  s3://swarm-floodrsct-data/manifests/mtbs_burn_perimeters/v2023/manifest.json

Filtered to: California (state_abbr = CA), years 2015-2023.
Full national file is ~80 MB; filtered CA output is ~5 MB.
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
log = logging.getLogger(__name__)

DST_BUCKET = "swarm-floodrsct-data"
DST_KEY = "raw/mtbs/perimeters/v2023/burn_perims_ca_2015_2023.parquet"
VERSION = "v2023"
DATASET = "mtbs_burn_perimeters"

# MTBS national perimeters — now hosted on USDA Forest Service FSGeodata Clearinghouse.
# Old edcintl.cr.usgs.gov URL is 404 as of May 2026.
MTBS_URL = (
    "https://data.fs.usda.gov/geodata/edw/edw_resources/shp/"
    "S_USA.MTBS_BURN_AREA_BOUNDARY.zip"
)

# Target bounding box: Riverside + San Bernardino + adjacent counties
TARGET_BBOX = {"minx": -118.5, "miny": 33.0, "maxx": -114.0, "maxy": 34.8}
TARGET_YEARS = range(2015, 2024)


def download_mtbs_zip(tmp_path: str) -> None:
    """Stream MTBS zip to a local tmp file (356 MB — avoid loading into RAM)."""
    log.info("Downloading MTBS national perimeters from %s ...", MTBS_URL)
    resp = requests.get(MTBS_URL, timeout=600, stream=True)
    resp.raise_for_status()
    total = 0
    with open(tmp_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            fh.write(chunk)
            total += len(chunk)
    log.info("MTBS download complete: %.1f MB", total / 1e6)


def filter_and_convert(zip_path: str) -> "gpd.GeoDataFrame":
    """Extract shapefile from ZIP, filter to CA + target years + bbox, return GeoDataFrame."""
    try:
        import geopandas as gpd
    except ImportError:
        log.error("geopandas required: pip install geopandas pyogrio")
        raise

    import pandas as pd
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(zip_path)
        zf.extractall(tmpdir)

        # Find the .shp file
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise RuntimeError("No .shp file found in MTBS zip")
        shp_path = shp_files[0]
        log.info("Reading shapefile: %s", shp_path.name)

        gdf = gpd.read_file(shp_path)
        log.info("MTBS national: %d perimeters, columns: %s", len(gdf), list(gdf.columns))

    # Standardize column names (MTBS schema: Ig_Date, Incid_Name, State, Incid_Type, ...)
    col_map = {}
    for col in gdf.columns:
        lc = col.lower()
        if "state" in lc:
            col_map[col] = "state_abbr"
        elif "ig_date" in lc or "ignition" in lc:
            col_map[col] = "ignition_date"
        elif "incid_name" in lc or "fire_name" in lc:
            col_map[col] = "fire_name"
        elif "incid_type" in lc:
            col_map[col] = "incident_type"
        elif "event_id" in lc:
            col_map[col] = "event_id"
        elif "asmnt_type" in lc:
            col_map[col] = "assessment_type"
    gdf = gdf.rename(columns=col_map)

    # Filter to California + target years
    if "state_abbr" in gdf.columns:
        gdf = gdf[gdf["state_abbr"] == "CA"].copy()
        log.info("CA fires: %d", len(gdf))

    if "ignition_date" in gdf.columns:
        gdf["ignition_date"] = pd.to_datetime(gdf["ignition_date"], errors="coerce")
        gdf["year"] = gdf["ignition_date"].dt.year
        gdf = gdf[gdf["year"].isin(TARGET_YEARS)].copy()
        log.info("CA fires 2015-2023: %d", len(gdf))

    # Further filter to bounding box
    gdf = gdf.to_crs("EPSG:4326")
    bb = TARGET_BBOX
    gdf = gdf.cx[bb["minx"]:bb["maxx"], bb["miny"]:bb["maxy"]].copy()
    log.info("CA fires in Riverside/SB area bbox: %d", len(gdf))

    return gdf


def main() -> None:
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        log.error("geopandas required: pip install geopandas")
        sys.exit(1)

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    try:
        s3.head_object(Bucket=DST_BUCKET, Key=DST_KEY)
        log.info("Already exists: s3://%s/%s — skipping", DST_BUCKET, DST_KEY)
        return
    except s3.exceptions.ClientError:
        pass

    # Import pandas here after confirming it's available
    import pandas as pd

    tmp_zip = "/tmp/mtbs_perims.zip"
    download_mtbs_zip(tmp_zip)
    checksum = hashlib.sha256(open(tmp_zip, "rb").read()).hexdigest()

    gdf = filter_and_convert(tmp_zip)
    if gdf.empty:
        log.warning("No MTBS perimeters found for target area — uploading empty parquet")

    local = "/tmp/burn_perims_ca_2015_2023.parquet"
    gdf.to_parquet(local, index=False)
    s3.upload_file(local, DST_BUCKET, DST_KEY)
    log.info("Uploaded s3://%s/%s (%d perimeters)", DST_BUCKET, DST_KEY, len(gdf))

    write_manifest(
        s3=s3,
        dataset=DATASET,
        version=VERSION,
        source_url=MTBS_URL,
        s3_key=DST_KEY,
        crs="EPSG:4326",
        record_count=len(gdf),
        sha256=checksum,
        license_="Public Domain — USGS/MTBS",
        notes=(
            f"MTBS burn perimeters filtered to CA, years 2015-2023, "
            f"bbox {TARGET_BBOX}. "
            "Binary ZCTA burn-scar overlap computed in build_event_dataset.py "
            "via build_burn_scar_features()."
        ),
    )

    log.info("fetch_mtbs_burn_scars complete")


if __name__ == "__main__":
    main()
