#!/usr/bin/env python3
"""
Bulk download FEMA NFHL county zips and upload to S3.
No extraction, no processing — just get the files.

Source: https://hazards.fema.gov/nfhlv2/output/County/{DFIRM_ID}_{DATE}.zip
Output: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/flood_raw/{dfirm_id}.zip
"""

import argparse
import gc
import json
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

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

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}


def _s3_upload(local_path: str, key: str):
    import boto3
    boto3.client("s3").upload_file(local_path, S3_BUCKET, key)


def _s3_exists_set() -> set:
    """List already-uploaded zips on S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        existing = set()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
            for obj in page.get("Contents", []):
                fname = obj["Key"].split("/")[-1]
                if fname.endswith(".zip"):
                    existing.add(fname[:-4])  # strip .zip
        return existing
    except Exception as e:
        log.warning("S3 list failed: %s", e)
        return set()


def download_one(entry: dict, work_dir: Path) -> tuple:
    """Download one county zip, upload to S3, delete local.

    Returns (dfirm_id, file_size_mb, error_msg).
    """
    dfirm_id = entry["dfirm_id"]
    url = entry["url"]
    zip_path = work_dir / f"{dfirm_id}.zip"

    try:
        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        _s3_upload(str(zip_path), f"{S3_RAW_PREFIX}{dfirm_id}.zip")
        return dfirm_id, size_mb, None

    except requests.HTTPError as e:
        return dfirm_id, 0, f"HTTP {e.response.status_code}"
    except Exception as e:
        return dfirm_id, 0, str(e)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="/opt/ml/processing/input/code/nfhl_download_catalog.json")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--conus-only", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="nfhl_"))

    log.info("Loading catalog: %s", args.catalog)
    with open(args.catalog) as f:
        catalog = json.load(f)
    log.info("  %d entries in catalog", len(catalog))

    if args.conus_only:
        catalog = [e for e in catalog if e["dfirm_id"][:2] not in NON_CONUS_PREFIXES]
        log.info("  CONUS filter: %d entries", len(catalog))

    # Dedup: keep only the latest version per county FIPS
    # Filename format: {DFIRM_ID}_{YYYYMMDD}.zip — extract date for sorting
    import re
    by_fips = {}
    for entry in catalog:
        fips = entry["dfirm_id"][:5]
        # Extract date from filename (e.g. "06001C_20250430.zip" -> "20250430")
        m = re.search(r"_(\d{8})\.", entry.get("filename", ""))
        entry_date = m.group(1) if m else "00000000"
        entry["_fips"] = fips
        entry["_date"] = entry_date
        if fips not in by_fips or entry_date > by_fips[fips]["_date"]:
            by_fips[fips] = entry
    deduped = list(by_fips.values())
    log.info("  Dedup: %d entries -> %d unique FIPS (latest version per county)",
             len(catalog), len(deduped))

    log.info("Checking S3 for existing uploads...")
    existing = _s3_exists_set()
    log.info("  Found %d existing on S3", len(existing))

    to_process = [e for e in deduped if e["dfirm_id"] not in existing]
    log.info("To download: %d (%d skipped)", len(to_process), len(deduped) - len(to_process))

    if not to_process:
        log.info("All counties already on S3. Done.")
        return

    log.info("=== DOWNLOADING %d COUNTY ZIPS (%d threads) ===", len(to_process), args.threads)
    t_start = time.time()
    done = 0
    failed = 0
    total_mb = 0.0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_one, e, work_dir): e for e in to_process}
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                dfirm_id, size_mb, error = fut.result()
                done += 1
                if error:
                    log.warning("  SKIP %s: %s", dfirm_id, error)
                    failed += 1
                else:
                    total_mb += size_mb

                if done % 50 == 0 or done == len(to_process):
                    elapsed = time.time() - t_start
                    rate = done / (elapsed / 60) if elapsed > 0 else 0
                    eta = (len(to_process) - done) / rate if rate > 0 else 0
                    log.info("  [%d/%d] %.0f MB total (%.0f/min, ETA %.0f min, %d failed)",
                             done, len(to_process), total_mb, rate, eta, failed)
                    sys.stdout.flush()
            except Exception as e:
                log.warning("  ERROR %s: %s", entry["dfirm_id"], e)
                failed += 1
                done += 1

    elapsed = time.time() - t_start
    log.info("=== DONE ===")
    log.info("  Downloaded: %d, Failed: %d, Total: %.1f MB",
             done - failed, failed, total_mb)
    log.info("  Elapsed: %.1f min", elapsed / 60)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "downloaded": done - failed,
        "failed": failed,
        "total_mb": round(total_mb, 1),
        "elapsed_sec": round(elapsed, 1),
    }
    summary_path = output_dir / "bulk_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _s3_upload(str(summary_path), f"{S3_RAW_PREFIX}bulk_download_summary.json")

    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
