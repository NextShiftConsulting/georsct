#!/usr/bin/env python3
"""
fetch_noaa_hrrr.py -- SageMaker job: download HRRR 3-km hourly QPF grids
for each FloodRSCT event and upload grib2 files to S3.

Source: NOMADS (recent) / Amazon S3 Requester-Pays (historical)
  - NOMADS: https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
  - AWS Open Data: s3://noaa-hrrr-bdp-pds/ (requester-pays, same account)
  - University of Utah HRRR archive: https://pando-rgw01.chpc.utah.edu/hrrr/

Output: s3://swarm-floodrsct-data/raw/noaa_hrrr/{event}/hrrr.tHHz.wrfsfcf01.grib2
  (1 file per initialization hour, forecast hour 01 = valid at T+1)

Variable extracted: APCP (total precipitation over forecast hour, kg m-2 = mm).

Usage:
    python fetch_noaa_hrrr.py --event harvey2017
    python fetch_noaa_hrrr.py --event imelda2019
    python fetch_noaa_hrrr.py --event beryl2024
    python fetch_noaa_hrrr.py --event ida2021_nola
    python fetch_noaa_hrrr.py --event ida2021_nyc
    python fetch_noaa_hrrr.py --event ian2022
    python fetch_noaa_hrrr.py --event helene2024
    python fetch_noaa_hrrr.py --event milton2024
    python fetch_noaa_hrrr.py --event hilary2023
    python fetch_noaa_hrrr.py --event ar_flood_2023

NOTE: HRRR was operational from 2014; all events are within archive range.
The University of Utah HRRR archive (pando-rgw01) is the primary source for
events older than ~2 months. For very recent events, NOMADS is used.

HRRR only covers CONUS; all FloodRSCT scenarios are within domain.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

from _s3_stream import s3_key_exists, stream_download_to_s3, stream_s3_object_to_tmp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DST_BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/noaa_hrrr"

# University of Utah HRRR archive — primary for historical events
UTAH_BASE = "https://pando-rgw01.chpc.utah.edu/hrrr/sfc"
# AWS Open Data — requester-pays, used as fallback
AWS_HRRR_BUCKET = "noaa-hrrr-bdp-pds"

# Events: start/end in UTC, forecast type suffix
EVENT_WINDOWS: dict[str, dict] = {
    "harvey2017": {
        "start": datetime(2017, 8, 17, 0, tzinfo=timezone.utc),
        "end": datetime(2017, 9, 4, 0, tzinfo=timezone.utc),
        "note": "Harvey landfall Aug 25; peak rainfall Aug 26-30",
    },
    "imelda2019": {
        "start": datetime(2019, 9, 17, 0, tzinfo=timezone.utc),
        "end": datetime(2019, 9, 22, 0, tzinfo=timezone.utc),
    },
    "beryl2024": {
        "start": datetime(2024, 7, 8, 0, tzinfo=timezone.utc),
        "end": datetime(2024, 7, 13, 0, tzinfo=timezone.utc),
    },
    "ida2021_nola": {
        "start": datetime(2021, 8, 26, 0, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 2, 0, tzinfo=timezone.utc),
    },
    "ida2021_nyc": {
        "start": datetime(2021, 9, 1, 0, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 4, 0, tzinfo=timezone.utc),
    },
    "ian2022": {
        "start": datetime(2022, 9, 23, 0, tzinfo=timezone.utc),
        "end": datetime(2022, 10, 2, 0, tzinfo=timezone.utc),
    },
    "helene2024": {
        "start": datetime(2024, 9, 24, 0, tzinfo=timezone.utc),
        "end": datetime(2024, 10, 2, 0, tzinfo=timezone.utc),
    },
    "milton2024": {
        "start": datetime(2024, 10, 7, 0, tzinfo=timezone.utc),
        "end": datetime(2024, 10, 13, 0, tzinfo=timezone.utc),
    },
    "hilary2023": {
        "start": datetime(2023, 8, 19, 0, tzinfo=timezone.utc),
        "end": datetime(2023, 8, 24, 0, tzinfo=timezone.utc),
    },
    "henri2021": {
        "start": datetime(2021, 8, 20, 0, tzinfo=timezone.utc),
        "end": datetime(2021, 8, 24, 0, tzinfo=timezone.utc),
    },
    "ar_flood_2023": {
        "start": datetime(2023, 3, 1, 0, tzinfo=timezone.utc),
        "end": datetime(2023, 3, 24, 0, tzinfo=timezone.utc),
        "note": "Atmospheric river series; ~22 days",
    },
}


def build_init_hours(start: datetime, end: datetime) -> list[datetime]:
    """All 6-hourly init times: 00Z, 06Z, 12Z, 18Z."""
    hours = []
    cur = start
    while cur < end:
        if cur.hour % 6 == 0:
            hours.append(cur)
        cur += timedelta(hours=6)
    return hours


def utah_url(dt: datetime, fhr: int = 1) -> str:
    """University of Utah HRRR archive URL for sfc product."""
    ymd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    return f"{UTAH_BASE}/{ymd}/hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2"


def aws_hrrr_key(dt: datetime, fhr: int = 1) -> str:
    """AWS Open Data HRRR key (requester-pays bucket)."""
    ymd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    return f"hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2"


def fetch_one(
    s3, dt: datetime, dst_bucket: str, dst_prefix: str, event: str
) -> bool:
    """Fetch one HRRR init time (fhr=01). Streams to disk → S3 multipart.

    s3=None uses thread-local client from _s3_stream.get_s3().
    """
    from _s3_stream import get_s3 as _get_s3
    client = s3 or _get_s3()

    fhr = 1
    hh = dt.strftime("%H")
    ymd = dt.strftime("%Y%m%d")
    dst_key = f"{dst_prefix}/{event}/hrrr.{ymd}.t{hh}z.wrfsfcf{fhr:02d}.grib2"

    # Try Utah archive first
    url = utah_url(dt, fhr)
    ok = stream_download_to_s3(client, url, dst_bucket, dst_key, timeout=180)
    if ok:
        return True

    # Fallback: AWS Open Data requester-pays
    import uuid as _uuid, os as _os
    aws_key = aws_hrrr_key(dt, fhr)
    log.info("Utah 404; trying AWS Open Data s3://%s/%s", AWS_HRRR_BUCKET, aws_key)
    tmp = f"/tmp/{_uuid.uuid4().hex}_hrrr.grib2"
    try:
        ok2 = stream_s3_object_to_tmp(client, AWS_HRRR_BUCKET, aws_key, tmp, request_payer=True)
        if not ok2:
            log.warning("AWS Open Data also failed for %s %s", event, dt.isoformat())
            return False
        client.upload_file(tmp, dst_bucket, dst_key)
        log.info("Uploaded (AWS fallback) s3://%s/%s", dst_bucket, dst_key)
        return True
    finally:
        if Path(tmp).exists():
            _os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        required=True,
        choices=list(EVENT_WINDOWS.keys()),
        help="Event window to fetch",
    )
    parser.add_argument(
        "--fhr",
        type=int,
        default=1,
        help="Forecast hour to pull (default 1 = T+1 hourly precip)",
    )
    args = parser.parse_args()

    event = args.event
    spec = EVENT_WINDOWS[event]
    init_hours = build_init_hours(spec["start"], spec["end"])

    workers = 16

    log.info(
        "Fetching HRRR for event=%s, %d init times (%s to %s) with %d workers",
        event,
        len(init_hours),
        spec["start"].date(),
        spec["end"].date(),
        workers,
    )

    ok = 0
    fail = 0
    # Pass s3=None — each worker thread gets its own boto3 client via _s3_stream.get_s3()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, None, dt, DST_BUCKET, DST_PREFIX, event): dt
            for dt in init_hours
        }
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                log.info("Progress: %d/%d (ok=%d fail=%d)", i, len(init_hours), ok, fail)
                sys.stdout.flush()

    log.info("Done. ok=%d fail=%d", ok, fail)
    if fail > 0:
        log.warning("%d init times failed — check Utah archive availability", fail)


if __name__ == "__main__":
    main()
