#!/usr/bin/env python3
"""
fetch_noaa_slosh.py -- SageMaker job: download NHC SLOSH retrospective surge outputs.

NHC publishes SLOSH Maximum Envelope of Water (MEOW) and Maximum of MEOWs (MOM)
grids for landfalling storms. Ian 2022, Helene 2024, and Milton 2024 all have
published SLOSH outputs.

Download source: https://www.nhc.noaa.gov/surge/slosh.php
Individual storm archives: https://www.nhc.noaa.gov/gis/archive_forecast_info_results.php

SLOSH grids are delivered as zipped shapefiles or NetCDF; we store as-is
and extract peak surge per ZCTA in build_event_dataset.py.

Outputs:
  s3://swarm-floodrsct-data/raw/noaa_slosh/ian2022/
  s3://swarm-floodrsct-data/raw/noaa_slosh/helene2024/
  s3://swarm-floodrsct-data/raw/noaa_slosh/milton2024/
"""

import logging
import re
import sys
import time
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
NHC_SURGE_BASE = "https://www.nhc.noaa.gov/surge"
NHC_GIS_ARCHIVE = "https://www.nhc.noaa.gov/gis/archive_forecast_info_results.php"

# NHC storm identifiers for SLOSH archive lookup
EVENTS = {
    "ian2022": {
        "nhc_id": "al092022",
        "name": "IAN",
        "year": 2022,
        "basin": "al",
    },
    "helene2024": {
        "nhc_id": "al092024",
        "name": "HELENE",
        "year": 2024,
        "basin": "al",
    },
    "milton2024": {
        "nhc_id": "al142024",
        "name": "MILTON",
        "year": 2024,
        "basin": "al",
    },
}

# Direct SLOSH product URLs (from NHC published archives — verify at runtime)
# Pattern: https://www.nhc.noaa.gov/storm_graphics/api/{NHCID}/refresh/AL092022_SLOSH_MOM.zip
SLOSH_URL_PATTERNS = [
    "https://www.nhc.noaa.gov/storm_graphics/api/{nhc_id_upper}/refresh/{nhc_id_upper}_SLOSH_MOM.zip",
    "https://www.nhc.noaa.gov/refresh/graphics_at{basin_num}+shtml/slosh.shtml",
    # Fallback: NHC GIS archive page for the storm
    "https://www.nhc.noaa.gov/gis/archive_forecast_info_results.php?id={nhc_id}&name={name}&year={year}",
]


def try_direct_download(event_name: str, event_cfg: dict) -> list[tuple[str, bytes]]:
    """Try direct SLOSH archive URL patterns. Return list of (filename, content) tuples."""
    results = []
    nhc_id_upper = event_cfg["nhc_id"].upper()
    basin_num = event_cfg["nhc_id"][2:4]  # e.g. "09" from "al092022"

    candidates = [
        f"https://www.nhc.noaa.gov/storm_graphics/api/{nhc_id_upper}/refresh/{nhc_id_upper}_SLOSH_MOM.zip",
        f"https://www.nhc.noaa.gov/storm_graphics/api/{nhc_id_upper}/refresh/{nhc_id_upper}_SLOSH_MEOW.zip",
    ]

    for url in candidates:
        log.info("Trying SLOSH URL: %s", url)
        resp = requests.get(url, timeout=300, stream=True)
        if resp.status_code == 200:
            filename = url.split("/")[-1]
            content = resp.content
            log.info("Downloaded %s (%d bytes)", filename, len(content))
            results.append((filename, content))
        else:
            log.debug("Not found (%d): %s", resp.status_code, url)
        time.sleep(0.5)

    return results


def scrape_gis_archive(event_cfg: dict) -> list[tuple[str, bytes]]:
    """Scrape NHC GIS archive page for SLOSH links."""
    results = []
    url = (
        f"{NHC_GIS_ARCHIVE}?id={event_cfg['nhc_id']}"
        f"&name={event_cfg['name']}&year={event_cfg['year']}"
    )
    log.info("Scraping NHC GIS archive: %s", url)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return results
        soup = BeautifulSoup(resp.text, "html.parser")
        slosh_links = [
            a["href"] for a in soup.find_all("a", href=True)
            if "slosh" in a["href"].lower() or "SLOSH" in a["href"]
        ]
        log.info("Found %d SLOSH links in GIS archive", len(slosh_links))
        for link in slosh_links[:10]:  # cap at 10 to avoid scraping everything
            full_url = link if link.startswith("http") else f"https://www.nhc.noaa.gov{link}"
            try:
                file_resp = requests.get(full_url, timeout=300, stream=True)
                if file_resp.status_code == 200:
                    filename = full_url.split("/")[-1].split("?")[0]
                    results.append((filename, file_resp.content))
                    log.info("Downloaded: %s (%d bytes)", filename, len(file_resp.content))
            except requests.RequestException as e:
                log.warning("Failed to download %s: %s", full_url, e)
            time.sleep(0.5)
    except Exception as e:
        log.warning("GIS archive scrape failed: %s", e)
    return results


def upload_files(files: list[tuple[str, bytes]], s3_prefix: str) -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    for filename, content in files:
        local_path = f"/tmp/{filename}"
        with open(local_path, "wb") as f:
            f.write(content)
        s3_key = f"{s3_prefix}/{filename}"
        s3.upload_file(local_path, BUCKET, s3_key)
        log.info("Uploaded to s3://%s/%s", BUCKET, s3_key)


def main() -> None:
    for event_name, event_cfg in EVENTS.items():
        log.info("Processing SLOSH data for %s", event_name)
        s3_prefix = f"raw/noaa_slosh/{event_name}"

        files = try_direct_download(event_name, event_cfg)
        if not files:
            log.info("Direct download failed; trying GIS archive scrape...")
            files = scrape_gis_archive(event_cfg)

        if not files:
            log.error(
                "No SLOSH files found for %s. Manual download required from "
                "https://www.nhc.noaa.gov/surge/slosh.php",
                event_name,
            )
            # Write a placeholder manifest so the job doesn't fail silently
            _aws = get_aws_credentials()
            _aws.pop("region_name", None)
            s3 = boto3.client("s3", region_name="us-east-1", **_aws)
            manifest = (
                f"SLOSH data not auto-downloaded for {event_name}.\n"
                f"Manual download required from https://www.nhc.noaa.gov/surge/slosh.php\n"
                f"Expected: {event_cfg['nhc_id'].upper()}_SLOSH_MOM.zip\n"
            ).encode()
            local = f"/tmp/{event_name}_slosh_manual_required.txt"
            with open(local, "wb") as f:
                f.write(manifest)
            s3.upload_file(local, BUCKET, f"{s3_prefix}/MANUAL_DOWNLOAD_REQUIRED.txt")
            continue

        upload_files(files, s3_prefix)

    log.info("fetch_noaa_slosh complete")


if __name__ == "__main__":
    main()
