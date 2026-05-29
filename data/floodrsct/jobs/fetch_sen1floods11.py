"""
fetch_sen1floods11.py -- SageMaker Processing job: download Sen1Floods11
benchmark metadata for flood segmentation comparison.

Sen1Floods11: harshinde/sen1floods11 (HuggingFace Datasets, 35.5 GB)
  11 flood events globally, SAR + optical imagery, pixel-level labels.
  Used as benchmark only — not a primary data source.

  Dataset structure: single sen1floods11.tar.gz (35.5 GB) + metadata GeoJSON.
  We download metadata only — the tar.gz is not needed for benchmark comparison.

  Confirmed sources (verified 2026-05-28):
    Metadata GeoJSON: harshinde/sen1floods11 (HF file repo, direct HTTPS)
      -> Sen1Floods11_Metadata.geojson (14 KB, 12 flood events)
    Full dataset:     gs://sen1floods11/ (Google Cloud Storage, public, 14 GB)
  NOTE: load_dataset() cannot be used — SageMaker image's conda-pinned `datasets`
  version predates the Mask feature type used in this dataset. Use direct HTTPS.

Role in the whitepaper:
  Benchmark for flood segmentation / Prithvi fine-tuning reference.
  Referenced in: Appendix -- Candidate datasets and benchmark comparisons.

Outputs to s3://swarm-floodrsct-data/raw/sen1floods11/
  Sen1Floods11_Metadata.geojson  -- 12-event flood metadata
  README.md                      -- dataset documentation
  summary.json                   -- download summary
  manifests/                     -- download manifest
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import requests

sys.path.insert(0, "/opt/ml/processing/input/code")
from _manifest_writer import write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
HF_BASE = "https://huggingface.co/datasets/harshinde/sen1floods11/resolve/main"
OUTPUT_PREFIX = "raw/sen1floods11"
MANIFEST_PREFIX = "manifests/sen1floods11/v1"

# Files to download (metadata only — no 35.5 GB tar.gz)
METADATA_FILES = [
    "Sen1Floods11_Metadata.geojson",
    "README.md",
]


def download_file(url: str, timeout: int = 60) -> bytes:
    """Download a file via HTTPS, following redirects."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    downloaded = []
    features = []

    for fname in METADATA_FILES:
        url = f"{HF_BASE}/{fname}"
        s3_key = f"{OUTPUT_PREFIX}/{fname}"

        # Skip if already in S3
        try:
            s3.head_object(Bucket=BUCKET, Key=s3_key)
            log.info("Already in S3, skipping: %s", fname)
            downloaded.append({"file": fname, "s3_key": s3_key, "skipped": True})
            continue
        except Exception:
            pass

        log.info("Downloading: %s", fname)
        content = download_file(url)
        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=content)
        size_kb = len(content) / 1024
        log.info("Uploaded %.1f KB to s3://%s/%s", size_kb, BUCKET, s3_key)
        downloaded.append({"file": fname, "s3_key": s3_key, "size_kb": round(size_kb, 1)})

        # Parse geojson for summary stats
        if fname.endswith(".geojson"):
            try:
                gj = json.loads(content)
                features = gj.get("features", [])
                countries = sorted({f["properties"].get("ISO_CC", "?")
                                    for f in features})
                log.info("GeoJSON: %d flood events, countries: %s",
                         len(features), countries)
            except Exception as e:
                log.warning("Could not parse geojson: %s", e)

    # Summary
    summary = {
        "dataset": "harshinde/sen1floods11",
        "hf_url": "https://huggingface.co/datasets/harshinde/sen1floods11",
        "benchmark_role": "flood_segmentation_reference",
        "whitepaper_placement": "appendix_benchmark_comparison",
        "n_flood_events": len(features),
        "files_downloaded": len([d for d in downloaded if not d.get("skipped")]),
        "files_skipped": len([d for d in downloaded if d.get("skipped")]),
        "note": (
            "Metadata only -- 35.5 GB sen1floods11.tar.gz not downloaded. "
            "GCS fallback: gs://sen1floods11/ (14 GB public bucket)."
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}/summary.json",
        Body=json.dumps(summary, indent=2).encode(),
        ContentType="application/json",
    )
    log.info("Summary: %d events, %d files", len(features), len(downloaded))

    # Manifest
    write_manifest(
        s3=s3,
        dataset="sen1floods11",
        version="v1",
        source_url="https://huggingface.co/datasets/harshinde/sen1floods11",
        s3_key=OUTPUT_PREFIX,
        record_count=len(features),
        license_="CC-BY-4.0",
        notes="metadata-only (geojson); 35.5GB tar.gz not downloaded; benchmark role only",
    )

    log.info("Done. Sen1Floods11 metadata at s3://%s/%s/", BUCKET, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
