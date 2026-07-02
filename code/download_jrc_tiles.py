"""
Download JRC Global Surface Water occurrence GeoTIFF tiles and upload to S3.
Tiles needed for FloodRSCT scenario areas.
"""

import boto3
import logging
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from swarm_auth import get_aws_credentials

BUCKET = "swarm-floodrsct-data"
S3_PREFIX = "raw/jrc_surface_water/v2021"
BASE_URL = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence"

TILES = [
    {
        "tile": "100W_40N",
        "areas": "Houston + New Orleans",
        "note": "100W-90W, 30N-40N",
    },
    {
        "tile": "80W_50N",
        "areas": "NYC",
        "note": "80W-70W, 40N-50N",
    },
    {
        "tile": "120W_40N",
        "areas": "Riverside-Coachella",
        "note": "120W-110W, 30N-40N",
    },
    {
        "tile": "90W_30N",
        "areas": "Southwest Florida",
        "note": "90W-80W, 20N-30N",
    },
]


def build_url(tile: str) -> str:
    return f"{BASE_URL}/occurrence_{tile}v1_4_2021.tif"


def build_s3_key(tile: str) -> str:
    return f"{S3_PREFIX}/occurrence_{tile}.tif"


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} TB"


def download_with_progress(url: str, dest_path: str) -> int:
    """Download url to dest_path, logging progress. Returns file size in bytes."""
    logger.info(f"  Downloading: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0 and block_num % 500 == 0 and block_num > 0:
            pct = min(100, downloaded * 100 / total_size)
            logger.info(
                f"    Progress: {human_size(downloaded)} / {human_size(total_size)} ({pct:.0f}%)"
            )

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    size = os.path.getsize(dest_path)
    return size


def upload_to_s3(s3_client, local_path: str, bucket: str, key: str) -> None:
    file_size = os.path.getsize(local_path)
    logger.info(f"  Uploading to s3://{bucket}/{key} ({human_size(file_size)})")

    # Use multipart for large files
    config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=50 * 1024 * 1024,  # 50 MB
        multipart_chunksize=50 * 1024 * 1024,
    )
    s3_client.upload_file(local_path, bucket, key, Config=config)
    logger.info(f"  Upload complete.")


def main():
    logger.info("Initializing S3 client via swarm_auth...")
    aws = get_aws_credentials()
    aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **aws)

    # Verify bucket is accessible
    try:
        s3.head_bucket(Bucket=BUCKET)
        logger.info(f"Bucket s3://{BUCKET} is accessible.")
    except Exception as e:
        logger.error(f"Cannot access bucket s3://{BUCKET}: {e}")
        sys.exit(1)

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for entry in TILES:
            tile = entry["tile"]
            areas = entry["areas"]
            note = entry["note"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Tile: {tile}  ({areas})")
            logger.info(f"  Coverage: {note}")

            url = build_url(tile)
            s3_key = build_s3_key(tile)
            local_path = os.path.join(tmpdir, f"occurrence_{tile}.tif")

            # Check if already uploaded
            try:
                head = s3.head_object(Bucket=BUCKET, Key=s3_key)
                existing_size = head["ContentLength"]
                logger.info(
                    f"  Already exists on S3 ({human_size(existing_size)}). Skipping download."
                )
                results.append(
                    {
                        "tile": tile,
                        "areas": areas,
                        "status": "SKIPPED (already on S3)",
                        "size": existing_size,
                        "s3_uri": f"s3://{BUCKET}/{s3_key}",
                    }
                )
                continue
            except s3.exceptions.ClientError:
                pass
            except Exception:
                pass

            try:
                t0 = time.time()
                file_size = download_with_progress(url, local_path)
                dl_time = time.time() - t0
                logger.info(
                    f"  Download complete: {human_size(file_size)} in {dl_time:.1f}s"
                )

                upload_to_s3(s3, local_path, BUCKET, s3_key)

                results.append(
                    {
                        "tile": tile,
                        "areas": areas,
                        "status": "OK",
                        "size": file_size,
                        "s3_uri": f"s3://{BUCKET}/{s3_key}",
                    }
                )

            except Exception as e:
                logger.error(f"  FAILED for tile {tile}: {e}")
                results.append(
                    {
                        "tile": tile,
                        "areas": areas,
                        "status": f"FAILED: {e}",
                        "size": 0,
                        "s3_uri": None,
                    }
                )

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        size_str = human_size(r["size"]) if r["size"] else "N/A"
        logger.info(
            f"  {r['tile']:15s} | {r['areas']:25s} | {size_str:10s} | {r['status']}"
        )
        if r["s3_uri"]:
            logger.info(f"    -> {r['s3_uri']}")

    failed = [r for r in results if r["status"].startswith("FAILED")]
    if failed:
        logger.error(f"\n{len(failed)} tile(s) failed.")
        sys.exit(1)
    else:
        logger.info("\nAll tiles uploaded successfully.")


if __name__ == "__main__":
    main()
