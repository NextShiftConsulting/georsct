"""
fetch_impactmesh_flood.py -- SageMaker Processing job: download ImpactMesh-Flood
benchmark data (masks + split lists only).

ImpactMesh-Flood: ibm-esa-geospatial/ImpactMesh-Flood (HuggingFace file repo)
  Multimodal flood extent benchmark dataset (IBM + ESA).
  Modalities: S1RTC, S2L2A, DEM, flood masks.
  Structure: train/val/test x {S1RTC.tar, S2L2A.tar, DEM.tar, MASK.tar} + split lists.
  License: CC-BY-4.0, not gated.

What we download (benchmark use only):
  - split/*.txt           (train/val/test scene lists, tiny)
  - {train,val,test}/MASK.tar  (flood mask rasters: ~307 MB total)
  - {train,val,test}/DEM.tar   (elevation context: ~333 MB total)
  - README.md
  We skip S1RTC (~9.3 GB test) and S2L2A (~29-109 GB) -- not needed for
  our benchmark comparison role.

Role in the whitepaper:
  Training data for TerraMind-base-Flood (the benchmark challenger EO expert).
  Referenced in: Section -- Model architecture / expert comparison.

Outputs to s3://swarm-floodrsct-data/raw/impactmesh_flood/
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
from swarm_auth import get_aws_credentials

sys.path.insert(0, "/opt/ml/processing/input/code")
from _manifest_writer import write_manifest
from _s3_stream import s3_key_exists, stream_download_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
HF_BASE = "https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood/resolve/main"
OUTPUT_PREFIX = "raw/impactmesh_flood"

# Files to download -- masks + DEMs + split lists only.
# S1RTC and S2L2A imagery tars are skipped (9-109 GB each, not needed for benchmark).
DOWNLOAD_FILES = [
    "README.md",
    "split/impactmesh_flood_train.txt",
    "split/impactmesh_flood_val.txt",
    "split/impactmesh_flood_test.txt",
    "train/MASK.tar",
    "val/MASK.tar",
    "test/MASK.tar",
    "train/DEM.tar",
    "val/DEM.tar",
    "test/DEM.tar",
]


def download_one(item: dict) -> dict:
    ok = stream_download_to_s3(
        None, item["url"], BUCKET, item["s3_key"], timeout=600, retries=3
    )
    return {"file": item["file"], "s3_key": item["s3_key"], "ok": ok}


def main() -> None:
    _aws = get_aws_credentials()
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    file_list = [
        {
            "file": f,
            "url": f"{HF_BASE}/{f}",
            "s3_key": f"{OUTPUT_PREFIX}/{f}",
        }
        for f in DOWNLOAD_FILES
    ]

    log.info("Downloading %d files (masks + DEMs + splits) from ImpactMesh-Flood",
             len(file_list))

    results = []
    n_ok = 0
    n_fail = 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_one, item): item for item in file_list}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if res["ok"]:
                n_ok += 1
                log.info("  [%d/%d] ok: %s", i, len(file_list), res["file"])
            else:
                n_fail += 1
                log.warning("  [%d/%d] FAIL: %s", i, len(file_list), res["file"])

    # Count split sizes from S3 objects
    split_counts = {}
    for split in ("train", "val", "test"):
        key = f"{OUTPUT_PREFIX}/split/impactmesh_flood_{split}.txt"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            lines = [l.strip() for l in obj["Body"].read().decode().splitlines() if l.strip()]
            split_counts[split] = len(lines)
        except Exception:
            split_counts[split] = None

    summary = {
        "dataset": "ibm-esa-geospatial/ImpactMesh-Flood",
        "hf_url": "https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood",
        "files_attempted": len(file_list),
        "files_ok": n_ok,
        "files_failed": n_fail,
        "downloaded": ["masks", "DEMs", "split_lists"],
        "skipped": ["S1RTC (9.3 GB test)", "S2L2A (29-109 GB per split)"],
        "split_scene_counts": split_counts,
        "benchmark_role": "terramind_flood_challenger_training_data",
        "whitepaper_placement": "section_model_architecture_expert_comparison",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}/summary.json",
        Body=json.dumps(summary, indent=2).encode(),
        ContentType="application/json",
    )

    write_manifest(
        s3=s3,
        dataset="impactmesh_flood",
        version="v1",
        source_url="https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood",
        s3_key=OUTPUT_PREFIX,
        record_count=sum(v for v in split_counts.values() if v),
        license_="CC-BY-4.0",
        notes="masks+DEMs+splits only; S1RTC+S2L2A imagery skipped; benchmark challenger role",
    )

    log.info("Summary: %d ok, %d failed -- splits: %s", n_ok, n_fail, split_counts)
    log.info("Done. ImpactMesh-Flood at s3://%s/%s/", BUCKET, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
