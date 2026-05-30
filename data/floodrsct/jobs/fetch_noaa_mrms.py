#!/usr/bin/env python3
"""
fetch_noaa_mrms.py -- SageMaker job: download NOAA MRMS Stage IV hourly precip.

Downloads Stage IV Multi-Sensor Precipitation Estimate (MPE) grib2 files from
NOAA's archive for the specified event window. Stage IV is gauge-corrected,
1-km resolution, hourly.

Archive: https://www.emc.ncep.noaa.gov/mmb/ylin/pcpanl/stage4/
         https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/prod/  (recent)
HDSS:    https://hdss.ncep.noaa.gov/  (historical archive access)

Files are stored as-is (grib2). ZCTA-level aggregation happens in
build_event_dataset.py using cfgrib + geopandas.

Output prefix: s3://swarm-floodrsct-data/raw/noaa_mrms/{event}/

WARNING: Harvey 2017 is ~17 days * 24 hrs = ~408 files, ~500 MB total.
Use ml.m5.xlarge for bandwidth headroom.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from _s3_stream import stream_download_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

# Stage IV archive base URLs (try in order; recent years on nomads, historical on EMC)
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/prod"
EMC_BASE = "https://www.emc.ncep.noaa.gov/mmb/ylin/pcpanl/stage4"

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
    "ida2021_nola": {
        "start": datetime(2021, 8, 26, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 2, tzinfo=timezone.utc),
    },
    "ida2021_nyc": {
        "start": datetime(2021, 9, 1, tzinfo=timezone.utc),
        "end": datetime(2021, 9, 4, tzinfo=timezone.utc),
    },
    # Southwest Florida
    "ian2022": {
        "start": datetime(2022, 9, 23, tzinfo=timezone.utc),
        "end": datetime(2022, 10, 1, tzinfo=timezone.utc),
    },
    "helene2024": {
        "start": datetime(2024, 9, 24, tzinfo=timezone.utc),
        "end": datetime(2024, 10, 1, tzinfo=timezone.utc),
    },
    "milton2024": {
        "start": datetime(2024, 10, 7, tzinfo=timezone.utc),
        "end": datetime(2024, 10, 12, tzinfo=timezone.utc),
    },
    # Riverside-Coachella
    "hilary2023": {
        "start": datetime(2023, 8, 19, tzinfo=timezone.utc),
        "end": datetime(2023, 8, 23, tzinfo=timezone.utc),
    },
    "ar_flood_2023": {
        "start": datetime(2023, 3, 1, tzinfo=timezone.utc),
        "end": datetime(2023, 3, 23, tzinfo=timezone.utc),
    },
}


def stage4_url_emc(dt: datetime) -> str:
    """EMC archive URL pattern for historical Stage IV hourly files."""
    # Pattern: stage4/YYYY/ST4.YYYYMMDDHHMMSS.01h.grb2
    return (
        f"{EMC_BASE}/{dt.year}/"
        f"ST4.{dt.strftime('%Y%m%d%H')}0000.01h.grb2"
    )


def stage4_url_nomads(dt: datetime) -> str:
    """NOMADS URL pattern (recent ~2 years)."""
    date_str = dt.strftime("%Y%m%d")
    return (
        f"{NOMADS_BASE}/pcpanl.{date_str}/"
        f"st4_conus.{dt.strftime('%Y%m%d%H')}.01h.grb2"
    )


_WORKERS = 16


def fetch_one_hour(s3, dt: datetime, event: str) -> tuple[datetime, bool]:
    """Fetch one Stage IV hourly file. Streams to disk → S3 multipart.

    s3=None uses thread-local client from _s3_stream.get_s3().
    """
    filename = f"stage4_{dt.strftime('%Y%m%d%H')}.grb2"
    s3_key = f"raw/noaa_mrms/{event}/{filename}"

    for url_fn in [stage4_url_nomads, stage4_url_emc]:
        url = url_fn(dt)
        ok = stream_download_to_s3(s3, url, BUCKET, s3_key, timeout=120)
        if ok:
            return dt, True

    log.warning("File not found for %s — gap in Stage IV archive", dt.isoformat())
    return dt, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, choices=list(EVENT_WINDOWS.keys()))
    args = parser.parse_args()

    window = EVENT_WINDOWS[args.event]
    hours: list[datetime] = []
    cur = window["start"]
    while cur <= window["end"]:
        hours.append(cur)
        cur += timedelta(hours=1)

    log.info("Fetching %d Stage IV hours for %s with %d workers", len(hours), args.event, _WORKERS)

    success_count = 0
    fail_count = 0

    # s3=None — each worker thread gets its own client via _s3_stream.get_s3()
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(fetch_one_hour, None, dt, args.event): dt for dt in hours}
        for i, fut in enumerate(as_completed(futures), 1):
            _, ok = fut.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
            if i % 25 == 0:
                log.info("Progress: %d/%d (ok=%d fail=%d)", i, len(hours), success_count, fail_count)
                sys.stdout.flush()

    total = success_count + fail_count
    log.info(
        "fetch_noaa_mrms complete for %s: %d/%d files downloaded, %d missing",
        args.event, success_count, total, fail_count,
    )
    if fail_count > total * 0.1:
        log.warning(
            "More than 10%% of files missing (%d/%d). "
            "Check HDSS portal for historical archive access: https://hdss.ncep.noaa.gov/",
            fail_count, total,
        )


if __name__ == "__main__":
    main()
