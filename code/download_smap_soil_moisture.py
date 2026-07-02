"""
Download SMAP L4 soil moisture (SPL4SMGP v008) granules for FloodRSCT storm events.

One snapshot per day for 7 days BEFORE each event (antecedent soil moisture).
Uploads to: s3://swarm-floodrsct-data/raw/smap_soil_moisture/v008/{event}/smap_sm_{YYYYMMDD}.h5

Events:
  harvey2017:  2017-08-18 to 2017-08-24
  ida2021:     2021-08-22 to 2021-08-28
  hilary2023:  2023-08-13 to 2023-08-19
  ian2022:     2022-09-21 to 2022-09-27
"""

import boto3
import logging
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CMR_URL = (
    "https://cmr.earthdata.nasa.gov/search/granules.json"
    "?short_name=SPL4SMGP&version=008"
    "&temporal[]={start}T00:00:00Z,{start}T03:00:00Z"
    "&page_size=1"
)

BEARER_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ"
    ".eyJ0eXBlIjoiVXNlciIsInVpZCI6InJlYWxydWR5bWFydGluIiwiZXhwIjoxNzg1Mjc5ODk2LCJpYXQiOjE3ODAwOTU4OTYsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30"
    ".GOglnVukgQji6NjN3Tkf5qrj3GK1F4ZtkFvaEkfNrPbz8-AjwSt0ZEn6UvPAw8vBo1W7sYcZruL5ZYZw7M8Rgi3mugLWUPpJwgTcPCg2XxOGK6Pq2yDZWpko6tHmR_Ggu1-Z3r3Hwji3G53UaDj3Ja3rAXM0EUSkrKt-T0XmkYgFCCMRi7WlQF7GpGXcZEpFQPdWiCy1rsdTB9asbFLroLfimTVhQNSqHXcmosZmHAwJwEAaRMegk29bhYbUbweRhRxGGOqrhGCOBkAmtpuvSuE0AobdzGmT5QOgGi0MS9p8ZOWk7GfK1Nj1F1WOynBpQWyoTPkqOdEsIGJTDubYYQ"
)

S3_BUCKET = "swarm-floodrsct-data"
S3_PREFIX = "raw/smap_soil_moisture/v008"

EVENTS = {
    "harvey2017": [date(2017, 8, 18) + timedelta(days=i) for i in range(7)],
    "ida2021":    [date(2021, 8, 22) + timedelta(days=i) for i in range(7)],
    "hilary2023": [date(2023, 8, 13) + timedelta(days=i) for i in range(7)],
    "ian2022":    [date(2022, 9, 21) + timedelta(days=i) for i in range(7)],
}


# ---------------------------------------------------------------------------
# HTTP session that preserves auth headers across cross-domain redirects
# ---------------------------------------------------------------------------
def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {BEARER_TOKEN}"})
    session.rebuild_auth = lambda prepared_request, response: None  # type: ignore[method-assign]
    return session


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------
def build_s3_client():
    from swarm_auth import get_aws_credentials
    aws = get_aws_credentials()
    aws.pop("region_name", None)
    return boto3.client("s3", region_name="us-east-1", **aws)


# ---------------------------------------------------------------------------
# CMR granule lookup
# ---------------------------------------------------------------------------
def find_granule_url(session: requests.Session, day: date) -> str | None:
    """Return the .h5 download URL for the first granule on *day*, or None."""
    url = CMR_URL.format(start=day.isoformat())
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    if not entries:
        log.warning("No CMR granule found for %s", day)
        return None
    links = entries[0].get("links", [])
    for link in links:
        rel = link.get("rel", "")
        href = link.get("href", "")
        if "data#" in rel and href.endswith(".h5"):
            return href
    # fallback: any .h5 link
    for link in links:
        if link.get("href", "").endswith(".h5"):
            return link["href"]
    log.warning("No .h5 link in granule for %s. Links: %s", day, links)
    return None


# ---------------------------------------------------------------------------
# S3 existence check
# ---------------------------------------------------------------------------
def s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Download + upload one file
# ---------------------------------------------------------------------------
def download_and_upload(
    session: requests.Session,
    s3,
    download_url: str,
    bucket: str,
    s3_key: str,
    day: date,
) -> int:
    """Download HDF5 to a temp file, stream-upload to S3. Returns file size in bytes."""
    log.info("  Downloading %s", download_url)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with session.get(download_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = 0
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
            log.info("  Downloaded %.1f MB -> %s", total / 1e6, tmp_path)

        log.info("  Uploading to s3://%s/%s", bucket, s3_key)
        s3.upload_file(
            tmp_path,
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/x-hdf5"},
        )
        log.info("  Upload complete.")
        return total
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    session = build_session()
    s3 = build_s3_client()

    total_files = 0
    event_sizes: dict[str, float] = {}

    for event, days in EVENTS.items():
        log.info("=== Event: %s (%d days) ===", event, len(days))
        event_bytes = 0.0

        for day in days:
            s3_key = f"{S3_PREFIX}/{event}/smap_sm_{day.strftime('%Y%m%d')}.h5"

            # Skip if already present
            if s3_key_exists(s3, S3_BUCKET, s3_key):
                log.info("  [SKIP] s3://%s/%s already exists", S3_BUCKET, s3_key)
                total_files += 1
                continue

            # Find granule URL
            download_url = find_granule_url(session, day)
            if download_url is None:
                log.error("  [ERROR] No granule URL for %s %s - skipping", event, day)
                continue

            # Download + upload
            try:
                nbytes = download_and_upload(
                    session, s3, download_url, S3_BUCKET, s3_key, day
                )
                event_bytes += nbytes
                total_files += 1
            except Exception as exc:
                log.error("  [ERROR] Failed for %s %s: %s", event, day, exc)

        event_sizes[event] = event_bytes
        log.info("  Event %s: %.1f MB downloaded/uploaded this run", event, event_bytes / 1e6)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("Total files processed: %d", total_files)
    for event, nbytes in event_sizes.items():
        log.info("  %-12s  %.1f MB new this run", event, nbytes / 1e6)


if __name__ == "__main__":
    main()
