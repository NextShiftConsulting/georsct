#!/usr/bin/env python3
"""
fetch_gpm_imerg_daily.py -- Download GPM IMERG Final Daily precipitation data
from NASA GES DISC for all 5 FloodRSCT storm events and upload to S3.

URL pattern:
  https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/
    {YYYY}/{DOY}/3B-DAY.MS.MRG.3IMERG.{YYYYMMDD}-S000000-E235959.V07B.nc4

Authentication: Bearer token via EARTHDATA_TOKEN env var.
GES DISC may redirect to an auth endpoint -- session handles redirects.

Output: s3://swarm-floodrsct-data/raw/gpm_imerg/daily/{event}/3B-DAY_{YYYYMMDD}.nc4
"""

import logging
import os
import sys
import tempfile
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import boto3
import requests
from swarm_auth import get_aws_credentials, get_credential

# Silence boto3 credential noise
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
GES_DISC_BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
MIN_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB -- reject HTML error pages

# Events: start/end inclusive (event window + 3 days antecedent already included)
EVENTS = {
    "harvey2017": {
        "start": date(2017, 8, 22),
        "end": date(2017, 9, 2),
    },
    "imelda2019": {
        "start": date(2019, 9, 14),
        "end": date(2019, 9, 21),
    },
    "beryl2024": {
        "start": date(2024, 7, 4),
        "end": date(2024, 7, 12),
    },
    "ida2021_nola": {
        "start": date(2021, 8, 26),
        "end": date(2021, 9, 1),
    },
    "ida2021_nyc": {
        "start": date(2021, 8, 29),
        "end": date(2021, 9, 4),
    },
    "henri2021": {
        "start": date(2021, 8, 18),
        "end": date(2021, 8, 24),
    },
    "hilary2023": {
        "start": date(2023, 8, 17),
        "end": date(2023, 8, 22),
    },
    "ian2022": {
        "start": date(2022, 9, 25),
        "end": date(2022, 9, 29),
    },
    "helene2024": {
        "start": date(2024, 9, 22),
        "end": date(2024, 9, 28),
    },
    "milton2024": {
        "start": date(2024, 10, 5),
        "end": date(2024, 10, 11),
    },
    "ar_flood_2023": {
        "start": date(2023, 1, 7),
        "end": date(2023, 1, 28),
    },
}


def build_url(d: date) -> str:
    """Build GES DISC URL for a given date."""
    yyyy = d.strftime("%Y")
    mm = d.strftime("%m")
    yyyymmdd = d.strftime("%Y%m%d")
    filename = f"3B-DAY.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4"
    return f"{GES_DISC_BASE}/{yyyy}/{mm}/{filename}"


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    """Return True if key already present in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False


def download_and_upload(
    session: requests.Session,
    s3_client,
    url: str,
    bucket: str,
    s3_key: str,
    retries: int = 3,
    timeout: int = 600,
) -> tuple[bool, int]:
    """Download a single IMERG file and upload to S3.

    Returns (success: bool, file_size_bytes: int).
    """
    if s3_key_exists(s3_client, bucket, s3_key):
        log.info("Already in S3, skipping: %s", s3_key)
        # Get existing file size from S3 head
        try:
            head = s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True, head.get("ContentLength", 0)
        except Exception:
            return True, 0

    tmp_name = f"{uuid.uuid4().hex}_imerg.nc4"
    tmp_path = Path(tempfile.gettempdir()) / tmp_name

    try:
        for attempt in range(retries):
            try:
                log.info("Downloading (attempt %d/%d): %s", attempt + 1, retries, url)
                resp = session.get(url, stream=True, timeout=timeout, allow_redirects=True)

                if resp.status_code == 404:
                    log.warning("404 Not Found: %s", url)
                    return False, 0

                if resp.status_code != 200:
                    log.warning(
                        "HTTP %d for %s -- body preview: %.200s",
                        resp.status_code,
                        url,
                        resp.text[:200],
                    )
                    if attempt < retries - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    return False, 0

                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                        fh.write(chunk)

                file_size = tmp_path.stat().st_size
                if file_size < MIN_SIZE_BYTES:
                    log.warning(
                        "File too small (%d bytes) -- likely an error page: %s",
                        file_size,
                        url,
                    )
                    # Show first 200 bytes as debug
                    with open(tmp_path, "rb") as fh:
                        preview = fh.read(200)
                    log.warning("Content preview: %s", preview[:200])
                    if attempt < retries - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    return False, 0

                size_mb = file_size / 1e6
                log.info("Uploading to s3://%s/%s (%.1f MB)", bucket, s3_key, size_mb)
                s3_client.upload_file(str(tmp_path), bucket, s3_key)
                log.info("Uploaded: s3://%s/%s", bucket, s3_key)
                return True, file_size

            except requests.RequestException as exc:
                log.warning("Request error attempt %d/%d: %s", attempt + 1, retries, exc)
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < retries - 1:
                    time.sleep(10 * (attempt + 1))

        return False, 0

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def date_range(start: date, end: date):
    """Yield dates from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_session(token: str) -> requests.Session:
    """Build a requests.Session with Bearer auth and redirect handling.

    GES DISC redirects to an Earthdata auth endpoint that requires the
    Authorization header to be forwarded on redirect. The default requests
    behavior strips auth headers on redirect to a different host, so we
    patch the session to always send the Authorization header.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    # Subclass to preserve Authorization header across redirects
    class AuthPreservingAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):
            # Ensure auth header is always set
            if "Authorization" not in request.headers:
                request.headers["Authorization"] = f"Bearer {token}"
            return super().send(request, **kwargs)

    # Hook to re-attach auth on redirect
    def _rebuild_auth(prepared_request, response):
        prepared_request.headers["Authorization"] = f"Bearer {token}"

    session.rebuild_auth = _rebuild_auth
    session.mount("https://", AuthPreservingAdapter())
    session.mount("http://", AuthPreservingAdapter())

    return session


def main():
    # Try env var -> swarm_auth -> direct Secrets Manager
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        token = (get_credential("EARTHDATA_TOKEN") or "").strip()
    if not token:
        # Direct Secrets Manager fallback (secret stored without swarm-it/ prefix)
        try:
            sm = boto3.client("secretsmanager", region_name="us-east-1")
            resp = sm.get_secret_value(SecretId="EARTHDATA_TOKEN")
            token = resp["SecretString"].strip()
        except Exception as e:
            log.warning("Secrets Manager fallback failed: %s", e)
    if not token:
        log.error(
            "EARTHDATA_TOKEN is not set. "
            "Please export it: export EARTHDATA_TOKEN=<your-token>  "
            "or add it to Secrets Manager as 'EARTHDATA_TOKEN'."
        )
        sys.exit(1)

    log.info("EARTHDATA_TOKEN present, length=%d", len(token))

    # Build S3 client via swarm_auth
    aws_creds = get_aws_credentials()
    aws_creds.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **aws_creds)
    log.info("S3 client ready (bucket=%s)", BUCKET)

    session = build_session(token)

    grand_total_files = 0
    grand_total_bytes = 0

    for event_name, window in EVENTS.items():
        log.info("=" * 60)
        log.info("Event: %s  (%s to %s)", event_name, window["start"], window["end"])
        log.info("=" * 60)

        event_files = 0
        event_bytes = 0
        event_failed = []

        days = list(date_range(window["start"], window["end"]))
        log.info("Fetching %d days for %s", len(days), event_name)

        for d in days:
            url = build_url(d)
            yyyymmdd = d.strftime("%Y%m%d")
            s3_key = f"raw/gpm_imerg/daily/{event_name}/3B-DAY_{yyyymmdd}.nc4"

            ok, size = download_and_upload(session, s3, url, BUCKET, s3_key)
            if ok:
                event_files += 1
                event_bytes += size
            else:
                event_failed.append(yyyymmdd)

        size_mb = event_bytes / 1e6
        log.info(
            "Event %s: %d files downloaded, %.1f MB total",
            event_name,
            event_files,
            size_mb,
        )
        if event_failed:
            log.warning("  Failed dates: %s", event_failed)

        grand_total_files += event_files
        grand_total_bytes += event_bytes

    log.info("=" * 60)
    log.info(
        "COMPLETE: %d total files, %.1f MB total across all events",
        grand_total_files,
        grand_total_bytes / 1e6,
    )

    # Per-event summary
    log.info("Per-event summary:")
    for event_name, window in EVENTS.items():
        days = list(date_range(window["start"], window["end"]))
        log.info("  %s: %d days", event_name, len(days))


if __name__ == "__main__":
    main()
