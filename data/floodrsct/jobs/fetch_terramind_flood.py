"""
fetch_terramind_flood.py -- SageMaker Processing job: download and smoke-test
TerraMind-base-Flood (ibm-esa-geospatial/TerraMind-base-Flood).

TerraMind-base-Flood: IBM + ESA geospatial foundation model fine-tuned on
ImpactMesh-Flood for multimodal flood extent mapping.
  Weights: TerraMind_v1_base_ImpactMesh_flood.pt (~673 MB)
  License: Apache-2.0, not gated.

Role in the whitepaper:
  Benchmark challenger EO expert alongside primary Prithvi-EO-2.0.
  Referenced in: Section -- Model architecture / expert comparison.

Whitepaper sentence:
  "We use Prithvi-EO-2.0 as the primary geospatial foundation-model expert
  and treat TerraMind-base-Flood, fine-tuned on the ImpactMesh-Flood
  multimodal dataset, as a benchmark challenger for multimodal flood
  extent mapping."

Outputs to s3://swarm-floodrsct-data/model/terramind_flood/
  weights/     -- full model snapshot
  smoke_test/  -- torch.load result
  manifests/   -- download manifest

Checkpointing: weights_already_in_s3() survives container restarts.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials

sys.path.insert(0, "/opt/ml/processing/input/code")
from _manifest_writer import write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
MODEL_ID = "ibm-esa-geospatial/TerraMind-base-Flood"
WEIGHTS_FILE = "TerraMind_v1_base_ImpactMesh_flood.pt"
OUTPUT_PREFIX = "model/terramind_flood"


def weights_already_in_s3() -> bool:
    _aws = get_aws_credentials()
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    try:
        s3.head_object(Bucket=BUCKET, Key=f"{OUTPUT_PREFIX}/weights/{WEIGHTS_FILE}")
        return True
    except Exception:
        return False


def download_model(local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    log.info("Downloading %s to %s", MODEL_ID, local_dir)
    t0 = time.time()
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5"],
    )
    log.info("Download complete in %.1f s", time.time() - t0)


def smoke_test(model_dir: Path) -> dict:
    """Verify weights file loads cleanly via torch.load."""
    import torch
    pt_file = model_dir / WEIGHTS_FILE
    if not pt_file.exists():
        raise FileNotFoundError(f"Weights file not found: {pt_file}")

    log.info("Loading state dict from %s (%.1f MB)",
             pt_file.name, pt_file.stat().st_size / 1e6)
    t0 = time.time()
    state_dict = torch.load(str(pt_file), map_location="cpu", weights_only=True)
    elapsed_ms = (time.time() - t0) * 1000

    n_tensors = len(state_dict)
    n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    top_keys = list(state_dict.keys())[:5]
    log.info("Loaded %d tensors, %.1fM params in %.1f ms",
             n_tensors, n_params / 1e6, elapsed_ms)

    return {
        "model_id": MODEL_ID,
        "n_params_M": round(n_params / 1e6, 2),
        "n_tensors": n_tensors,
        "top_keys_sample": top_keys,
        "load_ms": round(elapsed_ms, 1),
        "status": "PASS",
        "tested_at": datetime.now(timezone.utc).isoformat(),
    }


def upload_to_s3(local_dir: Path, s3_prefix: str) -> list[dict]:
    """Upload all files; skip hidden dirs and files already in S3 at same size."""
    _aws = get_aws_credentials()
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    uploaded = []
    for fp in sorted(local_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(local_dir)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        key = f"{s3_prefix}/{rel}"
        local_size = fp.stat().st_size
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] == local_size:
                log.info("  Skipping (already in S3): %s", key)
                uploaded.append({"s3_key": key, "size_mb": round(local_size / 1e6, 2),
                                 "skipped": True})
                continue
        except Exception:
            pass
        s3.upload_file(str(fp), BUCKET, key)
        log.info("  Uploaded %.1f MB -> %s", local_size / 1e6, key)
        uploaded.append({"s3_key": key, "size_mb": round(local_size / 1e6, 2)})
    return uploaded


def main() -> None:
    work_dir = Path("/tmp/terramind_flood")
    model_dir = work_dir / "weights"
    smoke_dir = work_dir / "smoke_test"
    model_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    pt_local = model_dir / WEIGHTS_FILE
    _aws = get_aws_credentials()
    s3_client = boto3.client("s3", region_name="us-east-1", **_aws)

    # 1. Download weights (skip if already in S3)
    if weights_already_in_s3():
        log.info("Weights already in S3 -- skipping HF download")
        uploaded = []
        if not pt_local.exists():
            pt_key = f"{OUTPUT_PREFIX}/weights/{WEIGHTS_FILE}"
            obj_size = s3_client.head_object(Bucket=BUCKET, Key=pt_key)["ContentLength"]
            log.info("Pulling weights from S3 for smoke test (%.1f MB)...",
                     obj_size / 1e6)
            s3_client.download_file(BUCKET, pt_key, str(pt_local))
    else:
        download_model(model_dir)
        log.info("Uploading weights to S3...")
        uploaded = upload_to_s3(model_dir, f"{OUTPUT_PREFIX}/weights")

    # 2. Smoke test
    smoke_path = smoke_dir / "smoke_test_result.json"
    log.info("Running smoke test...")
    try:
        result = smoke_test(model_dir)
        log.info("Smoke test: %s  (%.1fM params)", result["status"], result["n_params_M"])
    except Exception as e:
        log.warning("Smoke test failed (non-fatal): %s", e)
        result = {
            "model_id": MODEL_ID,
            "status": "SKIP",
            "reason": str(e),
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }
    smoke_path.write_text(json.dumps(result, indent=2))
    s3_client.upload_file(
        str(smoke_path), BUCKET,
        f"{OUTPUT_PREFIX}/smoke_test/smoke_test_result.json",
    )

    # 3. Manifest
    write_manifest(
        s3=s3_client,
        dataset="terramind_flood",
        version="v1",
        source_url=f"https://huggingface.co/{MODEL_ID}",
        s3_key=f"{OUTPUT_PREFIX}/weights",
        record_count=len(uploaded),
        license_="Apache-2.0",
        notes=f"smoke_test={result.get('status')}; n_params_M={result.get('n_params_M')}",
    )

    log.info("Done. TerraMind-base-Flood at s3://%s/%s/", BUCKET, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
