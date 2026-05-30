#!/usr/bin/env python3
"""
fetch_smap_soil_moisture.py -- Download SMAP L4 SPL4SMGP v008 antecedent
soil moisture for all 5 FloodRSCT storm events and upload to S3.

Strategy:
  1. Try NASA Harmony OGC bbox subsetting (Option A) -- produces small .nc4.
     If Harmony returns a non-200 or an async job, fall back to Option B.
  2. Fall back to direct NSIDC download of the full global HDF5 (~142 MB).

SMAP L4 SPL4SMGP v008 collection concept-id: C3480440870-NSIDC_CPRD

Antecedent window: 7 days BEFORE each event start (inclusive).
  - Harvey 2017:   2017-08-18 to 2017-08-24  (event start 2017-08-25)
  - Ida 2021:      2021-08-22 to 2021-08-28  (event start 2021-08-29, covers NOLA+NYC)
  - Hilary 2023:   2023-08-13 to 2023-08-19  (event start 2023-08-20)
  - Ian 2022:      2022-09-21 to 2022-09-27  (event start 2022-09-28)

S3 output:
  s3://swarm-floodrsct-data/raw/smap_soil_moisture/v008/{event}/smap_sm_{YYYYMMDD}.h5
  (or .nc4 if Harmony subsetting succeeds)

Authentication: Bearer token from EARTHDATA_TOKEN env var.
"""

import io
import json
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
S3_PREFIX = "raw/smap_soil_moisture/v008"

# SMAP L4 SPL4SMGP v008 collection concept-id at NSIDC CPRD
HARMONY_COLLECTION_ID = "C3480440870-NSIDC_CPRD"
HARMONY_BASE = "https://harmony.earthdata.nasa.gov"

# NSIDC direct download base
NSIDC_BASE = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/SMAP/SPL4SMGP/008"

# Minimum file size to accept as valid (rejects HTML error pages)
MIN_HARMONY_BYTES = 100 * 1024        # 100 KB for subsetted nc4
MIN_DIRECT_BYTES  = 50 * 1024 * 1024  # 50 MB for full HDF5

# Harmony async poll settings
HARMONY_POLL_INTERVAL = 15   # seconds between status polls
HARMONY_TIMEOUT       = 900  # seconds before giving up on async job

# Bounding boxes per event (lon_min, lat_min, lon_max, lat_max)
EVENT_BBOX = {
    "harvey2017":       (-96.5, 29.0, -94.5, 30.5),   # Houston metro
    "ida2021":          (-90.5, 29.5, -89.0, 30.5),   # NOLA core (antecedent window same for both sub-events)
    "hilary2023":       (-117.5, 33.0, -115.0, 34.5), # Riverside-Coachella
    "ian2022":          (-82.5, 25.5, -81.0, 27.5),   # SW Florida
}

# 7-day antecedent windows (inclusive start..end, 7 days each)
EVENTS = {
    "harvey2017":  {"start": date(2017, 8, 18), "end": date(2017, 8, 24)},
    "ida2021":     {"start": date(2021, 8, 22), "end": date(2021, 8, 28)},
    "hilary2023":  {"start": date(2023, 8, 13), "end": date(2023, 8, 19)},
    "ian2022":     {"start": date(2022, 9, 21), "end": date(2022, 9, 27)},
}


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def harmony_url(collection_id: str, bbox: tuple, d: date) -> str:
    """Build Harmony OGC subsetting URL for a single-day bbox extract.

    The time subset covers 00:00Z to 23:59:59Z so we capture the morning
    granule (~09:30 UTC overpass).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    date_str = d.strftime("%Y-%m-%d")
    time_start = f"{date_str}T00:00:00Z"
    time_end   = f"{date_str}T23:59:59Z"

    url = (
        f"{HARMONY_BASE}/{collection_id}/ogc-api-coverages/1.0.0"
        f"/collections/parameter_vars/coverage/rangeset"
        f"?subset=lat({lat_min}:{lat_max})"
        f"&subset=lon({lon_min}:{lon_max})"
        f'&subset=time("{time_start}":"{time_end}")'
        f"&format=application%2Fx-netcdf4"
    )
    return url


def nsidc_direct_url(d: date, hour: int = 9, version_tag: str = "Vv8011") -> tuple[str, str]:
    """Build NSIDC direct-download URL and expected filename.

    Returns (url, filename).  The version tag varies by processing epoch:
    Vv8011 for 2023+, Vv8010 for earlier data.  The path separator also
    changed from dots to slashes.
    """
    yyyy  = d.strftime("%Y")
    mm    = d.strftime("%m")
    dd    = d.strftime("%d")
    yyyymmdd = d.strftime("%Y%m%d")
    hh    = f"{hour:02d}"
    filename = f"SMAP_L4_SM_gph_{yyyymmdd}T{hh}3000_{version_tag}_001.h5"
    # Try slash-separated path (newer NSIDC layout)
    url = f"{NSIDC_BASE}/{yyyy}/{mm}/{dd}/{filename}"
    return url, filename


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def upload_file(s3_client, local_path: Path, bucket: str, key: str, size_mb: float) -> None:
    log.info("Uploading to s3://%s/%s (%.1f MB)", bucket, key, size_mb)
    s3_client.upload_file(str(local_path), bucket, key)
    log.info("Uploaded: s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# Auth-preserving HTTP session
# ---------------------------------------------------------------------------

def build_session(token: str) -> requests.Session:
    """Build a requests.Session that forwards the Bearer token on redirects."""
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    class _AuthAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):
            request.headers.setdefault("Authorization", f"Bearer {token}")
            return super().send(request, **kwargs)

    def _rebuild_auth(prepared_request, response):
        prepared_request.headers["Authorization"] = f"Bearer {token}"

    session.rebuild_auth = _rebuild_auth
    session.mount("https://", _AuthAdapter())
    session.mount("http://",  _AuthAdapter())
    return session


# ---------------------------------------------------------------------------
# Harmony Option A
# ---------------------------------------------------------------------------

def _poll_harmony_job(session: requests.Session, job_url: str) -> str | None:
    """Poll a Harmony async job until it has a download link or times out.

    Returns the first result URL (netCDF4 file) or None on failure/timeout.
    """
    deadline = time.time() + HARMONY_TIMEOUT
    while time.time() < deadline:
        resp = session.get(job_url, timeout=60)
        if resp.status_code != 200:
            log.warning("Harmony job poll HTTP %d: %s", resp.status_code, job_url)
            return None
        try:
            body = resp.json()
        except Exception:
            log.warning("Harmony job poll non-JSON response")
            return None

        status = body.get("status", "").lower()
        log.info("  Harmony job status: %s  progress: %s%%", status, body.get("progress", "?"))

        if status == "successful":
            links = body.get("links", [])
            for lnk in links:
                href = lnk.get("href", "")
                rel  = lnk.get("rel", "")
                if "data" in rel or href.endswith(".nc4") or href.endswith(".nc"):
                    return href
            # Fallback: try first link with type data
            for lnk in links:
                if lnk.get("type", "").startswith("application/"):
                    return lnk.get("href")
            log.warning("Harmony job successful but no data link found in response")
            return None

        if status in ("failed", "canceled"):
            log.warning("Harmony job %s: %s", status, body.get("message", ""))
            return None

        time.sleep(HARMONY_POLL_INTERVAL)

    log.warning("Harmony job timed out after %d s", HARMONY_TIMEOUT)
    return None


def try_harmony(
    session: requests.Session,
    s3_client,
    event_name: str,
    d: date,
    retries: int = 2,
) -> tuple[bool, int]:
    """Attempt Harmony subsetting for one day. Returns (success, bytes_uploaded)."""
    bbox  = EVENT_BBOX[event_name]
    url   = harmony_url(HARMONY_COLLECTION_ID, bbox, d)
    yyyymmdd = d.strftime("%Y%m%d")
    s3_key = f"{S3_PREFIX}/{event_name}/smap_sm_{yyyymmdd}.nc4"

    if s3_key_exists(s3_client, BUCKET, s3_key):
        head = s3_client.head_object(Bucket=BUCKET, Key=s3_key)
        sz = head.get("ContentLength", 0)
        log.info("Already in S3 (Harmony .nc4), skipping: %s  (%d bytes)", s3_key, sz)
        return True, sz

    # Also check if the full-file .h5 was already uploaded (skip re-download)
    s3_key_h5 = f"{S3_PREFIX}/{event_name}/smap_sm_{yyyymmdd}.h5"
    if s3_key_exists(s3_client, BUCKET, s3_key_h5):
        head = s3_client.head_object(Bucket=BUCKET, Key=s3_key_h5)
        sz = head.get("ContentLength", 0)
        log.info("Full-file .h5 already in S3, skipping Harmony: %s  (%d bytes)", s3_key_h5, sz)
        return True, sz

    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_smap.nc4"
    try:
        for attempt in range(retries):
            try:
                log.info(
                    "Harmony attempt %d/%d for %s %s",
                    attempt + 1, retries, event_name, yyyymmdd,
                )
                log.debug("  URL: %s", url)
                resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
                log.info("  Harmony HTTP %d (Content-Type: %s)", resp.status_code, resp.headers.get("Content-Type", "?"))

                if resp.status_code == 202:
                    # Async job -- poll for completion
                    try:
                        body = resp.json()
                    except Exception:
                        log.warning("  Harmony 202 but non-JSON body; treating as failure")
                        break

                    job_url = body.get("links", [{}])[0].get("href") or body.get("href")
                    # Find the status link more carefully
                    for lnk in body.get("links", []):
                        if lnk.get("rel") in ("self", "status"):
                            job_url = lnk.get("href")
                            break
                    if not job_url:
                        # Use the Harmony jobs endpoint
                        job_id = body.get("jobID")
                        if job_id:
                            job_url = f"{HARMONY_BASE}/jobs/{job_id}"

                    if not job_url:
                        log.warning("  Harmony async: cannot determine job URL from response")
                        break

                    log.info("  Harmony async job, polling: %s", job_url)
                    download_url = _poll_harmony_job(session, job_url)
                    if not download_url:
                        log.warning("  Harmony async job did not complete successfully")
                        break

                    log.info("  Harmony async result URL: %s", download_url)
                    dl_resp = session.get(download_url, stream=True, timeout=600)
                    if dl_resp.status_code != 200:
                        log.warning("  Harmony result download HTTP %d", dl_resp.status_code)
                        break

                    with open(tmp_path, "wb") as fh:
                        for chunk in dl_resp.iter_content(chunk_size=2 * 1024 * 1024):
                            fh.write(chunk)

                elif resp.status_code == 200:
                    # Synchronous response
                    ct = resp.headers.get("Content-Type", "")
                    if "html" in ct or "json" in ct:
                        # Might be a redirect to an async job description
                        preview = resp.content[:500]
                        log.warning("  Harmony 200 but Content-Type=%s; preview: %s", ct, preview[:200])
                        # Try to parse as JSON job response
                        try:
                            body = resp.json()
                            if "jobID" in body or "status" in body:
                                job_id = body.get("jobID")
                                job_url = f"{HARMONY_BASE}/jobs/{job_id}" if job_id else None
                                if job_url:
                                    log.info("  Treating as async job: %s", job_url)
                                    download_url = _poll_harmony_job(session, job_url)
                                    if not download_url:
                                        break
                                    dl_resp = session.get(download_url, stream=True, timeout=600)
                                    if dl_resp.status_code != 200:
                                        break
                                    with open(tmp_path, "wb") as fh:
                                        for chunk in dl_resp.iter_content(chunk_size=2 * 1024 * 1024):
                                            fh.write(chunk)
                                else:
                                    break
                        except Exception:
                            break
                    else:
                        with open(tmp_path, "wb") as fh:
                            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                                fh.write(chunk)

                elif resp.status_code in (400, 404, 422):
                    log.warning(
                        "  Harmony %d (unrecoverable) for %s: %s",
                        resp.status_code, yyyymmdd, resp.text[:300],
                    )
                    return False, 0

                else:
                    log.warning("  Harmony HTTP %d for %s", resp.status_code, yyyymmdd)
                    if attempt < retries - 1:
                        time.sleep(20 * (attempt + 1))
                    continue

                # Validate file
                if not tmp_path.exists() or tmp_path.stat().st_size < MIN_HARMONY_BYTES:
                    sz = tmp_path.stat().st_size if tmp_path.exists() else 0
                    log.warning(
                        "  Harmony file too small (%d bytes) for %s -- rejecting",
                        sz, yyyymmdd,
                    )
                    if tmp_path.exists():
                        tmp_path.unlink()
                    if attempt < retries - 1:
                        time.sleep(20)
                    continue

                file_size = tmp_path.stat().st_size
                upload_file(s3_client, tmp_path, BUCKET, s3_key, file_size / 1e6)
                return True, file_size

            except requests.RequestException as exc:
                log.warning("  Harmony request error attempt %d: %s", attempt + 1, exc)
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < retries - 1:
                    time.sleep(20)

        return False, 0

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Direct NSIDC download Option B
# ---------------------------------------------------------------------------

def try_direct(
    session: requests.Session,
    s3_client,
    event_name: str,
    d: date,
    retries: int = 3,
) -> tuple[bool, int]:
    """Fall back: download full NSIDC HDF5 granule (~142 MB)."""
    yyyymmdd = d.strftime("%Y%m%d")
    s3_key = f"{S3_PREFIX}/{event_name}/smap_sm_{yyyymmdd}.h5"

    if s3_key_exists(s3_client, BUCKET, s3_key):
        head = s3_client.head_object(Bucket=BUCKET, Key=s3_key)
        sz = head.get("ContentLength", 0)
        log.info("Already in S3 (direct .h5), skipping: %s  (%.1f MB)", s3_key, sz / 1e6)
        return True, sz

    # Try both version tags (Vv8011 for 2023+, Vv8010 for earlier)
    version_tags = ["Vv8011", "Vv8010"] if d.year >= 2023 else ["Vv8010", "Vv8011"]
    url = filename = None
    for vtag in version_tags:
        url, filename = nsidc_direct_url(d, hour=9, version_tag=vtag)
        log.info("Direct NSIDC download: %s", url)
        probe = session.head(url, timeout=30, allow_redirects=True)
        if probe.status_code == 200:
            log.info("  Version tag %s found", vtag)
            break
        log.info("  Version tag %s -> HTTP %d, trying next", vtag, probe.status_code)
    else:
        # Use first version tag as default
        url, filename = nsidc_direct_url(d, hour=9, version_tag=version_tags[0])

    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{filename}"
    try:
        for attempt in range(retries):
            try:
                log.info("  Direct attempt %d/%d", attempt + 1, retries)
                resp = session.get(url, stream=True, timeout=900, allow_redirects=True)
                log.info("  HTTP %d  Content-Type: %s", resp.status_code, resp.headers.get("Content-Type", "?"))

                if resp.status_code == 404:
                    # Try alternate hours and version tags
                    found = False
                    for vtag in version_tags:
                        for alt_hour in [1, 4, 6, 12, 16, 19, 22]:
                            alt_url, alt_filename = nsidc_direct_url(d, hour=alt_hour, version_tag=vtag)
                            log.info("  Trying %s hour %02d:30 UTC", vtag, alt_hour)
                            alt_resp = session.get(alt_url, stream=True, timeout=900, allow_redirects=True)
                            if alt_resp.status_code == 200:
                                resp = alt_resp
                                url = alt_url
                                filename = alt_filename
                                log.info("  Found granule at %s hour %02d", vtag, alt_hour)
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        log.warning("  No granule found for %s on any hour/version", yyyymmdd)
                        return False, 0

                if resp.status_code != 200:
                    log.warning(
                        "  HTTP %d for %s  body preview: %.200s",
                        resp.status_code, yyyymmdd, resp.text[:200],
                    )
                    if attempt < retries - 1:
                        time.sleep(30 * (attempt + 1))
                    continue

                with open(tmp_path, "wb") as fh:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (20 * 1024 * 1024) < 8 * 1024 * 1024:
                            log.info("    Downloaded %.0f MB so far...", downloaded / 1e6)

                file_size = tmp_path.stat().st_size
                if file_size < MIN_DIRECT_BYTES:
                    log.warning(
                        "  Direct file too small (%d bytes) for %s -- likely not an HDF5",
                        file_size, yyyymmdd,
                    )
                    # Show content preview
                    with open(tmp_path, "rb") as fh:
                        preview = fh.read(200)
                    log.warning("  Content preview: %s", preview)
                    tmp_path.unlink()
                    if attempt < retries - 1:
                        time.sleep(30)
                    continue

                upload_file(s3_client, tmp_path, BUCKET, s3_key, file_size / 1e6)
                return True, file_size

            except requests.RequestException as exc:
                log.warning("  Direct request error attempt %d: %s", attempt + 1, exc)
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < retries - 1:
                    time.sleep(30)

        return False, 0

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    # Token: env var first, then swarm_auth
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        token = (get_credential("EARTHDATA_TOKEN") or "").strip()
    if not token:
        log.error(
            "EARTHDATA_TOKEN is not set. "
            "Export it: export EARTHDATA_TOKEN=<your-token>"
        )
        sys.exit(1)
    log.info("EARTHDATA_TOKEN present, length=%d", len(token))

    aws_creds = get_aws_credentials()
    aws_creds.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **aws_creds)
    log.info("S3 client ready (bucket=%s)", BUCKET)

    session = build_session(token)

    # Probe Harmony with a quick HEAD to check service availability
    harmony_ok = True
    try:
        probe = session.get(f"{HARMONY_BASE}/", timeout=15)
        if probe.status_code >= 500:
            log.warning("Harmony probe returned %d -- will skip Harmony and go direct", probe.status_code)
            harmony_ok = False
        else:
            log.info("Harmony probe: HTTP %d -- service reachable", probe.status_code)
    except Exception as exc:
        log.warning("Harmony probe failed: %s -- will skip Harmony", exc)
        harmony_ok = False

    grand_total_files  = 0
    grand_total_bytes  = 0
    grand_total_failed = []

    for event_name, window in EVENTS.items():
        log.info("=" * 70)
        log.info(
            "Event: %-20s  antecedent window: %s to %s",
            event_name, window["start"], window["end"],
        )
        bbox = EVENT_BBOX[event_name]
        log.info("  bbox (lon_min, lat_min, lon_max, lat_max): %s", bbox)
        log.info("=" * 70)

        event_files  = 0
        event_bytes  = 0
        event_failed = []

        days = list(date_range(window["start"], window["end"]))
        log.info("Fetching %d days for %s", len(days), event_name)

        for d in days:
            yyyymmdd = d.strftime("%Y%m%d")
            ok = False
            sz = 0

            if harmony_ok:
                log.info("--- %s  Option A (Harmony) ---", yyyymmdd)
                ok, sz = try_harmony(session, s3, event_name, d)

            if not ok:
                if harmony_ok:
                    log.info("--- %s  Harmony failed, falling back to Option B (direct NSIDC) ---", yyyymmdd)
                else:
                    log.info("--- %s  Option B (direct NSIDC) ---", yyyymmdd)
                ok, sz = try_direct(session, s3, event_name, d)

            if ok:
                event_files += 1
                event_bytes += sz
                log.info("  OK  %s  %.2f MB", yyyymmdd, sz / 1e6)
            else:
                event_failed.append(yyyymmdd)
                log.error("  FAIL  %s", yyyymmdd)

        log.info(
            "Event %s: %d/%d files, %.1f MB",
            event_name, event_files, len(days), event_bytes / 1e6,
        )
        if event_failed:
            log.warning("  Failed dates: %s", event_failed)
            grand_total_failed.extend([f"{event_name}/{d}" for d in event_failed])

        grand_total_files += event_files
        grand_total_bytes += event_bytes

    log.info("=" * 70)
    log.info(
        "COMPLETE: %d total files, %.1f MB total across all events",
        grand_total_files, grand_total_bytes / 1e6,
    )

    if grand_total_failed:
        log.warning("FAILED dates (%d):", len(grand_total_failed))
        for item in grand_total_failed:
            log.warning("  %s", item)
    else:
        log.info("All dates downloaded successfully.")

    # Per-event summary
    log.info("Per-event summary:")
    for event_name, window in EVENTS.items():
        days = list(date_range(window["start"], window["end"]))
        log.info("  %-20s  %d days  (s3://%s/%s/%s/)", event_name, len(days), BUCKET, S3_PREFIX, event_name)


if __name__ == "__main__":
    main()
