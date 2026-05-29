"""
fetch_noaa_mrms_v2.py -- Fetch NOAA MRMS Stage IV hourly precipitation grids.

Sources (priority order):
  1. Iowa State Mesonet archive (PRIMARY) -- full historical coverage
  2. EMC NCEP archive (FALLBACK) -- sometimes returns error pages with HTTP 200

Guards:
  - Minimum 10 KB file size check (real GRIB2 files are 100+ KB)
  - Pre-fetch pass deletes existing S3 stubs < 10 KB from prior failed runs
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Container-local imports
sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest
from _s3_stream import stream_download_to_s3, s3_key_exists, get_s3

import boto3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/noaa_mrms"
MIN_FILE_SIZE = 10 * 1024  # 10 KB -- real GRIB2 files are 100+ KB
MAX_WORKERS = 16

EVENT_WINDOWS = {
    "harvey2017": {
        "start": datetime(2017, 8, 17, tzinfo=timezone.utc),
        "end": datetime(2017, 9, 4, tzinfo=timezone.utc),
    },
    "imelda2019": {
        "start": datetime(2019, 9, 17, tzinfo=timezone.utc),
        "end": datetime(2019, 9, 22, tzinfo=timezone.utc),
    },
    "beryl2024": {
        "start": datetime(2024, 7, 7, tzinfo=timezone.utc),
        "end": datetime(2024, 7, 13, tzinfo=timezone.utc),
    },
    "ida2021_nyc": {
        "start": datetime(2021, 9, 1, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 4, tzinfo=timezone.utc),
    },
    "ian2022": {
        "start": datetime(2022, 9, 23, tzinfo=timezone.utc),
        "end": datetime(2022, 10, 1, tzinfo=timezone.utc),
    },
    "hilary2023": {
        "start": datetime(2023, 8, 19, tzinfo=timezone.utc),
        "end": datetime(2023, 8, 23, tzinfo=timezone.utc),
    },
}

# URL templates
# Primary: Iowa State Mesonet GaugeCorr QPE 01H (gzip-compressed GRIB2, ~700KB each)
IOWA_URL = (
    "https://mtarchive.geol.iastate.edu/"
    "{year:04d}/{month:02d}/{day:02d}/mrms/ncep/GaugeCorr_QPE_01H/"
    "GaugeCorr_QPE_01H_00.00_{year:04d}{month:02d}{day:02d}-{hour:02d}0000.grib2.gz"
)
# Fallback: EMC NCEP Stage IV archive
EMC_URL = (
    "https://www.emc.ncep.noaa.gov/mmb/ylin/pcpanl/stage4/"
    "{year:04d}/ST4.{year:04d}{month:02d}{day:02d}{hour:02d}0000.01h.grb2"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hourly_timestamps(start: datetime, end: datetime):
    """Yield hourly datetimes from start up to (but not including) end."""
    dt = start
    while dt < end:
        yield dt
        dt += timedelta(hours=1)


def s3_key_for(event: str, dt: datetime) -> str:
    """Build the S3 key for a given hour."""
    fname = f"GaugeCorr_QPE_01H_00.00_{dt:%Y%m%d}-{dt:%H}0000.grib2.gz"
    return f"{DST_PREFIX}/{event}/{fname}"


def urls_for(dt: datetime) -> list:
    """Return [primary, fallback] URLs for a given hour."""
    parts = {
        "year": dt.year, "month": dt.month,
        "day": dt.day, "hour": dt.hour,
    }
    return [IOWA_URL.format(**parts), EMC_URL.format(**parts)]


def delete_s3_stubs(s3_client, event: str, timestamps: list) -> int:
    """Delete existing S3 objects < MIN_FILE_SIZE for this event.

    Returns the number of stubs deleted.
    """
    deleted = 0
    for dt in timestamps:
        key = s3_key_for(event, dt)
        try:
            resp = s3_client.head_object(Bucket=BUCKET, Key=key)
            size = resp["ContentLength"]
            if size < MIN_FILE_SIZE:
                s3_client.delete_object(Bucket=BUCKET, Key=key)
                log.info(
                    "Deleted stub (%d bytes): s3://%s/%s", size, BUCKET, key
                )
                deleted += 1
        except s3_client.exceptions.ClientError:
            # Key does not exist -- nothing to delete
            pass
    return deleted


def fetch_one_hour(event: str, dt: datetime) -> bool:
    """Try to fetch one hourly GRIB2 file using primary then fallback URL.

    Returns True if the file is now in S3 (either freshly uploaded or
    already present from a prior run).
    """
    key = s3_key_for(event, dt)
    s3 = get_s3()

    # Already present and valid (stub cleanup ran earlier)
    if s3_key_exists(s3, BUCKET, key):
        return True

    for url in urls_for(dt):
        ok = stream_download_to_s3(
            s3=s3,
            url=url,
            bucket=BUCKET,
            key=key,
            min_size_bytes=MIN_FILE_SIZE,
        )
        if ok:
            return True

    log.warning("ALL sources failed for %s", dt.isoformat())
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch NOAA MRMS Stage IV hourly precip to S3"
    )
    parser.add_argument(
        "--event", required=True, choices=list(EVENT_WINDOWS.keys()),
        help="Storm event key",
    )
    args = parser.parse_args()

    event = args.event
    window = EVENT_WINDOWS[event]
    timestamps = list(hourly_timestamps(window["start"], window["end"]))
    total = len(timestamps)

    log.info(
        "Event=%s | %s to %s | %d hourly files",
        event, window["start"].isoformat(), window["end"].isoformat(), total,
    )

    # ------------------------------------------------------------------
    # Pass 0: delete existing stub files (< 10 KB) from prior bad runs
    # ------------------------------------------------------------------
    s3_main = boto3.client("s3", region_name="us-east-1")
    deleted = delete_s3_stubs(s3_main, event, timestamps)
    log.info("Stub cleanup: deleted %d files < %d bytes", deleted, MIN_FILE_SIZE)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Pass 1: fetch in parallel
    # ------------------------------------------------------------------
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_one_hour, event, dt): dt
            for dt in timestamps
        }
        for i, future in enumerate(as_completed(futures), 1):
            dt = futures[future]
            try:
                ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception:
                log.exception("Unhandled error for %s", dt.isoformat())
                failed += 1

            if i % 25 == 0 or i == total:
                log.info(
                    "Progress: %d/%d (ok=%d, fail=%d)",
                    i, total, success, failed,
                )
                sys.stdout.flush()

    log.info(
        "Fetch complete: %d/%d succeeded, %d failed", success, total, failed
    )

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    write_manifest(
        s3=s3_main,
        dataset="noaa_mrms",
        version=f"{event}_v2",
        source_url="https://mtarchive.geol.iastate.edu (primary), EMC (fallback)",
        s3_key=f"{DST_PREFIX}/{event}/",
        crs="EPSG:4326",
        record_count=success,
        notes=(
            f"Stage IV 01h GRIB2, {event}, "
            f"{window['start']:%Y-%m-%d} to {window['end']:%Y-%m-%d}. "
            f"min_size_check={MIN_FILE_SIZE}B. "
            f"{success}/{total} files fetched."
        ),
    )

    if failed > 0:
        log.warning("%d/%d files could not be fetched", failed, total)
        sys.exit(1)


if __name__ == "__main__":
    main()
