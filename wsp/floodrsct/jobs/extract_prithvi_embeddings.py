"""
extract_prithvi_embeddings.py -- SageMaker Processing job: extract per-ZCTA
embeddings from Prithvi-EO-2.0 using HLS satellite imagery.

Pipeline:
    1. Load ZCTA centroids from scenario event_features parquet
    2. Query NASA LP DAAC CMR for HLS tiles covering each centroid
    3. Download 6-band HLS chips (224x224 at 30m resolution)
    4. Run Prithvi encoder (no masking) on batches
    5. Mean-pool patch tokens (1024-dim) as per-ZCTA embedding
    6. Quality validation (norms, cosine discrimination, NaN/Inf)
    7. Upload embeddings parquet + metadata JSON to S3

Inputs:
    s3://swarm-floodrsct-data/processed/{scenario}/{scenario}_event_features.parquet
    s3://swarm-floodrsct-data/model/prithvi_eo2/weights/

Outputs:
    s3://swarm-floodrsct-data/results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet
    s3://swarm-floodrsct-data/results/s035/prithvi_embeddings/{scenario}_prithvi_meta.json

Usage (SageMaker only -- no local runs):
    Launched by scripts/launch_extract_prithvi_embeddings.py
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

import boto3
from swarm_auth import get_aws_credentials

sys.path.insert(0, "/opt/ml/processing/input/code")
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
MODEL_PREFIX = "model/prithvi_eo2/weights"
OUTPUT_PREFIX = "results/s035/prithvi_embeddings"

# Scenario name -> S3 event_features prefix (abbreviated names on S3)
SCENARIO_PREFIX = {
    "houston": "houston",
    "nyc": "nyc",
    "new_orleans": "no",
    "riverside_coachella": "rc",
    "southwest_florida": "swfl",
}

# Prithvi-EO-2.0 config (from config.json on S3)
PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
CHIP_SIZE = 224  # pixels
HLS_RESOLUTION = 30  # meters per pixel
# 224 * 30m = 6720m ~ 6.7km chip footprint

# NASA CMR API for HLS
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
HLS_COLLECTION_L30 = "C2021957657-LPCLOUD"  # HLS Landsat
HLS_COLLECTION_S30 = "C2021957295-LPCLOUD"  # HLS Sentinel-2

# Band name mapping: HLS Sentinel-2 -> Prithvi index
S30_BAND_MAP = {
    "B02": 0,  # Blue
    "B03": 1,  # Green
    "B04": 2,  # Red
    "B8A": 3,  # Narrow NIR (B05 in Prithvi = B8A in S30)
    "B11": 4,  # SWIR1 (B06 in Prithvi = B11 in S30)
    "B12": 5,  # SWIR2 (B07 in Prithvi = B12 in S30)
}

# Fallback: ZCTAs without HLS data get NaN embeddings (not zeros).
# Running zero chips through the encoder produces valid-looking vectors
# that are indistinguishable from real HLS embeddings by norm or cosine.


# ---------------------------------------------------------------------------
# HLS tile discovery and download
# ---------------------------------------------------------------------------

CMR_MAX_RETRIES = 3
CMR_BACKOFF = [5, 15, 30]  # seconds between retries


def find_hls_granule(lat: float, lon: float, max_cloud: int = 30) -> dict | None:
    """Find a recent, low-cloud HLS granule covering the given lat/lon.

    Searches Sentinel-2 HLS (S30) first (more frequent), falls back to
    Landsat HLS (L30). Retries transient failures with backoff.

    Returns dict with granule_id and download links, or None if not found.
    """
    point_str = f"{lon},{lat}"

    for collection_id, label in [
        (HLS_COLLECTION_S30, "S30"),
        (HLS_COLLECTION_L30, "L30"),
    ]:
        params = {
            "collection_concept_id": collection_id,
            "point": point_str,
            "temporal": "2023-01-01T00:00:00Z,2024-12-31T23:59:59Z",
            "cloud_cover": f"0,{max_cloud}",
            "sort_key": "-start_date",
            "page_size": 5,
        }

        entries = None
        for attempt in range(CMR_MAX_RETRIES):
            try:
                resp = requests.get(CMR_URL, params=params, timeout=30)
                resp.raise_for_status()
                entries = resp.json().get("feed", {}).get("entry", [])
                break
            except Exception as e:
                if attempt < CMR_MAX_RETRIES - 1:
                    wait = CMR_BACKOFF[attempt]
                    log.warning("CMR retry %d/%d for (%.4f, %.4f) %s: %s (wait %ds)",
                                attempt + 1, CMR_MAX_RETRIES, lat, lon, label, e, wait)
                    time.sleep(wait)
                else:
                    log.warning("CMR exhausted retries for (%.4f, %.4f) %s: %s",
                                lat, lon, label, e)

        if not entries:
            continue

        # Pick the first (most recent, lowest cloud) granule
        entry = entries[0]
        granule_id = entry.get("producer_granule_id", entry.get("title", "unknown"))
        links = [
            link["href"] for link in entry.get("links", [])
            if link.get("href", "").endswith(".tif")
        ]

        return {
            "granule_id": granule_id,
            "collection": label,
            "links": links,
            "cloud_cover": entry.get("cloud_cover"),
        }

    return None


def download_hls_chip(
    granule: dict,
    lat: float,
    lon: float,
    work_dir: Path,
    earthdata_token: str | None = None,
) -> np.ndarray | None:
    """Download 6-band HLS chip centered on (lat, lon).

    Returns (6, 224, 224) float32 array normalized by Prithvi stats,
    or None on failure.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    band_files = {}
    needed_bands = S30_BAND_MAP if granule["collection"] == "S30" else {
        "B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4, "B07": 5,
    }

    headers = {}
    if earthdata_token:
        headers["Authorization"] = f"Bearer {earthdata_token}"

    for link in granule["links"]:
        for band_name in needed_bands:
            if f".{band_name}." in link or link.endswith(f"_{band_name}.tif"):
                local_path = work_dir / f"{band_name}.tif"
                if not local_path.exists():
                    try:
                        r = requests.get(link, headers=headers, timeout=120,
                                         allow_redirects=True)
                        r.raise_for_status()
                        local_path.write_bytes(r.content)
                    except Exception as e:
                        log.warning("Failed to download %s: %s", link, e)
                        return None
                band_files[band_name] = local_path

    if len(band_files) < len(needed_bands):
        log.warning("Only got %d/%d bands for granule %s",
                    len(band_files), len(needed_bands), granule["granule_id"])
        return None

    # Read and extract chip centered on (lat, lon)
    chip = np.zeros((6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    half_m = (CHIP_SIZE * HLS_RESOLUTION) / 2  # half chip in meters

    for band_name, prithvi_idx in needed_bands.items():
        with rasterio.open(band_files[band_name]) as src:
            # Transform lat/lon to the raster's CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x_center, y_center = transformer.transform(lon, lat)

            # Window centered on the point
            window = from_bounds(
                x_center - half_m, y_center - half_m,
                x_center + half_m, y_center + half_m,
                src.transform,
            )

            try:
                data = src.read(
                    1,
                    window=window,
                    out_shape=(CHIP_SIZE, CHIP_SIZE),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                chip[prithvi_idx] = data.astype(np.float32)
            except Exception as e:
                log.warning("Failed to read window for %s: %s", band_name, e)
                return None

    # Reject chips where >50% of pixels are nodata (value 0 pre-normalization)
    nodata_frac = (chip == 0).mean()
    if nodata_frac > 0.5:
        log.warning("Chip for granule %s has %.0f%% nodata pixels -- rejecting",
                    granule["granule_id"], nodata_frac * 100)
        return None

    # Normalize using Prithvi stats
    mean = np.array(PRITHVI_MEAN, dtype=np.float32).reshape(6, 1, 1)
    std = np.array(PRITHVI_STD, dtype=np.float32).reshape(6, 1, 1)
    chip = (chip - mean) / std

    return chip


# ---------------------------------------------------------------------------
# Prithvi model loading and inference
# ---------------------------------------------------------------------------

def load_prithvi_encoder(model_dir: Path, device: str = "cpu") -> torch.nn.Module:
    """Load Prithvi-EO-2.0 encoder (PrithviViT) from state dict.

    The full PrithviMAE has encoder + decoder. We only need the encoder
    for embedding extraction.
    """
    # Add model dir to path so prithvi_mae module is importable
    sys.path.insert(0, str(model_dir))
    from prithvi_mae import PrithviViT

    config = json.loads((model_dir / "config.json").read_text())
    cfg = config["pretrained_cfg"]

    encoder = PrithviViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        num_frames=cfg["num_frames"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        coords_encoding=cfg.get("coords_encoding"),
        coords_scale_learn=cfg.get("coords_scale_learn", False),
    )

    # Load weights from full MAE checkpoint -- extract encoder keys
    weights_path = model_dir / "Prithvi_EO_V2_300M_TL.pt"
    full_state = torch.load(str(weights_path), map_location=device, weights_only=True)

    # Filter to encoder keys only (prefix "encoder.")
    encoder_state = {}
    for k, v in full_state.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    if encoder_state:
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        log.info("Loaded encoder weights: %d keys, %d missing, %d unexpected",
                 len(encoder_state), len(missing), len(unexpected))
    else:
        # Keys may not have "encoder." prefix -- try direct load
        missing, unexpected = encoder.load_state_dict(full_state, strict=False)
        log.info("Loaded weights directly: %d missing, %d unexpected",
                 len(missing), len(unexpected))

    encoder.eval()
    encoder.to(device)
    return encoder


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    chips: np.ndarray,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Run Prithvi encoder on (N, 6, 224, 224) chips, return (N, 1024) embeddings.

    Uses forward with mask_ratio=0 (no masking) to get full sequence,
    then mean-pools patch tokens (excluding CLS) for spatial discrimination.
    """
    n = len(chips)
    embed_dim = encoder.embed_dim
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(chips[start:end]).to(device)
        # Prithvi expects 5D: (B, C, T, H, W) -- add time dim for single frame
        batch = batch.unsqueeze(2)

        # forward with mask_ratio=0: returns (x, mask, ids_restore)
        # x shape: (batch, 1 + num_patches, embed_dim) -- CLS at position 0
        x, _, _ = encoder(batch, mask_ratio=0.0)

        # Mean-pool patch tokens (skip CLS at position 0).
        # CLS token is dominated by architectural bias (cosine > 0.99 for all
        # pairs). Patch token mean captures spatial land-cover variation.
        patch_emb = x[:, 1:, :].mean(dim=1).cpu().numpy()
        embeddings[start:end] = patch_emb

        if (start // batch_size) % 10 == 0:
            log.info("  Encoded %d/%d chips", end, n)

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        help="Scenario name (houston, nyc, etc.)")
    parser.add_argument("--max-cloud", type=int, default=30,
                        help="Max cloud cover %% for HLS query")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--max-zctas", type=int, default=0,
                        help="Limit ZCTAs for testing (0=all)")
    args = parser.parse_args()

    scenario = args.scenario
    work_dir = Path("/tmp/prithvi_extract")
    model_dir = work_dir / "model"
    chips_dir = work_dir / "chips"
    model_dir.mkdir(parents=True, exist_ok=True)
    chips_dir.mkdir(parents=True, exist_ok=True)

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    # ---------------------------------------------------------------
    # 1. Load ZCTA centroids
    # ---------------------------------------------------------------
    s3_prefix = SCENARIO_PREFIX.get(scenario, scenario)
    features_key = f"processed/{scenario}/{s3_prefix}_event_features.parquet"
    local_features = work_dir / "event_features.parquet"
    log.info("Downloading %s", features_key)
    s3.download_file(BUCKET, features_key, str(local_features))
    df = pd.read_parquet(local_features)

    # Deduplicate to unique ZCTAs (event_features has one row per ZCTA-event)
    zcta_col = "zcta_id"
    if zcta_col not in df.columns:
        log.error("ABORT: expected column '%s' not found. Columns: %s",
                  zcta_col, list(df.columns[:10]))
        sys.exit(1)

    zctas = df.drop_duplicates(subset=[zcta_col])[[zcta_col, "latitude", "longitude"]].copy()
    zctas = zctas.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    if args.max_zctas > 0:
        zctas = zctas.head(args.max_zctas)
        log.info("Limited to %d ZCTAs for testing", len(zctas))

    n_zctas = len(zctas)
    log.info("Scenario %s: %d unique ZCTAs with coordinates", scenario, n_zctas)

    # ---------------------------------------------------------------
    # 2. Download Prithvi model weights
    # ---------------------------------------------------------------
    for fname in ["Prithvi_EO_V2_300M_TL.pt", "config.json", "prithvi_mae.py"]:
        local_path = model_dir / fname
        if not local_path.exists():
            s3_key = f"{MODEL_PREFIX}/{fname}"
            log.info("Downloading %s", s3_key)
            s3.download_file(BUCKET, s3_key, str(local_path))

    # ---------------------------------------------------------------
    # 3. Fetch HLS chips for each ZCTA
    # ---------------------------------------------------------------
    # Try env var first, then Secrets Manager directly.
    # AWSSecretsAdapter defaults to "swarm-it/" prefix but EARTHDATA_TOKEN
    # is stored as a bare key -- bypass the adapter.
    earthdata_token = os.environ.get("EARTHDATA_TOKEN")
    if not earthdata_token:
        try:
            _sm = boto3.client("secretsmanager", region_name="us-east-1", **_aws)
            earthdata_token = _sm.get_secret_value(
                SecretId="EARTHDATA_TOKEN"
            )["SecretString"]
        except Exception as e:
            log.warning("Could not retrieve EARTHDATA_TOKEN from Secrets Manager: %s", e)
    if earthdata_token:
        log.info("Earthdata token found -- HLS downloads will use Bearer auth")
    else:
        log.warning("No EARTHDATA_TOKEN -- HLS downloads may fail (403)")

    log.info("Fetching HLS chips for %d ZCTAs (max_cloud=%d%%)", n_zctas, args.max_cloud)

    chips = np.zeros((n_zctas, 6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hls_meta = []
    fallback_indices = []  # track which rows are fallback
    n_hls_ok = 0
    n_fallback = 0

    for idx, (i, row) in enumerate(zctas.iterrows()):
        zcta_id = str(row[zcta_col])
        lat, lon = float(row["latitude"]), float(row["longitude"])
        zcta_chip_dir = chips_dir / zcta_id
        zcta_chip_dir.mkdir(exist_ok=True)

        granule = find_hls_granule(lat, lon, max_cloud=args.max_cloud)
        if granule is not None:
            chip = download_hls_chip(granule, lat, lon, zcta_chip_dir,
                                     earthdata_token=earthdata_token)
            if chip is not None:
                chips[idx] = chip
                n_hls_ok += 1
                hls_meta.append({
                    "zcta": zcta_id,
                    "source": "hls",
                    "granule_id": granule["granule_id"],
                    "collection": granule["collection"],
                    "cloud_cover": granule.get("cloud_cover"),
                })
                if (n_hls_ok + n_fallback) % 25 == 0:
                    log.info("  Progress: %d/%d ZCTAs (%d HLS, %d fallback)",
                             n_hls_ok + n_fallback, n_zctas, n_hls_ok, n_fallback)
                continue

        # Fallback: leave chip as zeros (will be replaced with NaN embeddings)
        n_fallback += 1
        fallback_indices.append(idx)
        hls_meta.append({
            "zcta": zcta_id,
            "source": "fallback_no_data",
            "granule_id": None,
            "collection": None,
            "cloud_cover": None,
        })

    log.info("HLS fetch complete: %d OK, %d fallback, %d total",
             n_hls_ok, n_fallback, n_zctas)

    fallback_pct = 100 * n_fallback / max(n_zctas, 1)
    if fallback_pct > 80:
        log.error("ABORT: %.1f%% fallback (%d/%d ZCTAs). "
                  "Embeddings would be meaningless. Check EARTHDATA_TOKEN and HLS availability.",
                  fallback_pct, n_fallback, n_zctas)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 4. Load Prithvi encoder and extract embeddings
    # ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Prithvi encoder on %s", device)
    encoder = load_prithvi_encoder(model_dir, device=device)

    log.info("Extracting embeddings (batch_size=%d)", args.batch_size)
    t0 = time.time()
    embeddings = extract_embeddings(encoder, chips, batch_size=args.batch_size, device=device)
    elapsed = time.time() - t0
    log.info("Embedding extraction complete: %d ZCTAs, %.1f s (%.2f s/ZCTA)",
             n_zctas, elapsed, elapsed / max(n_zctas, 1))

    # Set fallback rows to NaN so downstream can distinguish them.
    # Running zero chips through the encoder produces valid-looking vectors
    # (norm ~23.7, cosine ~0.99 with real HLS) that poison downstream models.
    if fallback_indices:
        embeddings[fallback_indices] = np.nan
        log.info("Set %d fallback rows to NaN (indices: %s...)",
                 len(fallback_indices),
                 str(fallback_indices[:5]))

    # ---------------------------------------------------------------
    # 5. Quality validation
    # ---------------------------------------------------------------
    hls_mask = np.ones(n_zctas, dtype=bool)
    hls_mask[fallback_indices] = False
    hls_emb = embeddings[hls_mask]

    n_nan = int(np.isnan(hls_emb).any(axis=1).sum())
    n_inf = int(np.isinf(hls_emb).any(axis=1).sum())
    if n_nan > 0 or n_inf > 0:
        log.error("QUALITY: %d NaN rows, %d Inf rows in HLS embeddings", n_nan, n_inf)

    norms = np.linalg.norm(hls_emb, axis=1)
    log.info("QUALITY: HLS embedding norms: min=%.2f, max=%.2f, mean=%.2f, std=%.4f",
             norms.min(), norms.max(), norms.mean(), norms.std())

    # Check for near-duplicate embeddings (discrimination)
    if len(hls_emb) > 1:
        normed = hls_emb / (norms[:, None] + 1e-10)
        # Sample pairwise cosine (full matrix too expensive for large N)
        sample_n = min(50, len(normed))
        sample = normed[:sample_n]
        cos_matrix = sample @ sample.T
        upper = cos_matrix[np.triu_indices(sample_n, k=1)]
        log.info("QUALITY: pairwise cosine (n=%d): min=%.4f, max=%.4f, mean=%.4f",
                 sample_n, upper.min(), upper.max(), upper.mean())

    # ---------------------------------------------------------------
    # 6. Build output dataframe and upload
    # ---------------------------------------------------------------
    # Embedding columns: prithvi_emb_0 ... prithvi_emb_1023
    emb_cols = [f"prithvi_emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df.insert(0, "zcta", zctas[zcta_col].values)

    # Add HLS source metadata
    meta_df = pd.DataFrame(hls_meta)
    out_df = emb_df.merge(meta_df[["zcta", "source"]], on="zcta", how="left")

    # Save locally and upload
    out_parquet = work_dir / f"{scenario}_prithvi_embeddings.parquet"
    out_df.to_parquet(out_parquet, index=False)
    log.info("Saved embeddings: %s (%d rows, %d cols)",
             out_parquet.name, len(out_df), len(out_df.columns))

    emb_s3_key = f"{OUTPUT_PREFIX}/{scenario}_prithvi_embeddings.parquet"
    s3.upload_file(str(out_parquet), BUCKET, emb_s3_key)
    log.info("Uploaded s3://%s/%s", BUCKET, emb_s3_key)

    # Upload metadata JSON
    meta_out = {
        "scenario": scenario,
        "n_zctas": n_zctas,
        "n_hls_ok": n_hls_ok,
        "n_fallback": n_fallback,
        "fallback_pct": round(100 * n_fallback / max(n_zctas, 1), 1),
        "embed_dim": int(embeddings.shape[1]),
        "pooling": "mean_patch",
        "model": "Prithvi-EO-2.0-300M-TL",
        "device": device,
        "extraction_time_s": round(elapsed, 1),
        "max_cloud_cover": args.max_cloud,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality": {
            "hls_norm_min": round(float(norms.min()), 2) if len(norms) > 0 else None,
            "hls_norm_max": round(float(norms.max()), 2) if len(norms) > 0 else None,
            "hls_norm_mean": round(float(norms.mean()), 2) if len(norms) > 0 else None,
            "n_nan_hls_rows": n_nan,
            "n_inf_hls_rows": n_inf,
            "fallback_value": "NaN",
        },
        "hls_coverage": [m for m in hls_meta if m["source"] == "hls"],
    }
    meta_s3_key = f"{OUTPUT_PREFIX}/{scenario}_prithvi_meta.json"
    upload_json_result(s3, BUCKET, meta_s3_key, meta_out)

    log.info("Done. Outputs at s3://%s/%s/", BUCKET, OUTPUT_PREFIX)
    log.info("Coverage: %.1f%% HLS, %.1f%% fallback",
             100 * n_hls_ok / max(n_zctas, 1),
             100 * n_fallback / max(n_zctas, 1))


if __name__ == "__main__":
    main()
