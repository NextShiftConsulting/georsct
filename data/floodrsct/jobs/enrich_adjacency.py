#!/usr/bin/env python3
"""enrich_adjacency.py -- Add degree and centroid distance to adjacency parquet.

Reads the existing zcta_adjacency.parquet (edge list) and zcta5_boundaries.parquet
(polygons), computes:
  - degree: number of neighbors per ZCTA (added to both sides of each edge)
  - centroid_distance_km: haversine distance between ZCTA centroids

Writes enriched parquet back to S3 (archives original first).

Usage:
    python enrich_adjacency.py --upload
    python enrich_adjacency.py --dry-run
"""

import argparse
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

ADJ_KEY = "raw/geocertdb2026/zcta_adjacency.parquet"
BOUNDS_KEY = "raw/geocertdb2026/zcta5_boundaries.parquet"


def _load_parquet(s3, key: str) -> pd.DataFrame:
    """Load parquet from S3."""
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_centroids(bounds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute centroids from WKB geometry column."""
    import shapely

    log.info("Computing centroids for %d ZCTAs...", len(bounds_df))
    geoms = shapely.from_wkb(bounds_df["geometry"])
    centroids = shapely.centroid(geoms)
    return pd.DataFrame({
        "zcta_id": bounds_df["zcta_id"].astype(str),
        "centroid_lat": shapely.get_y(centroids),
        "centroid_lon": shapely.get_x(centroids),
    })


def main():
    parser = argparse.ArgumentParser(description="Enrich adjacency with degree + distance")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would enrich %s with degree + centroid_distance_km", ADJ_KEY)
        log.info("Reads: %s, %s", ADJ_KEY, BOUNDS_KEY)
        log.info("Writes: %s (archived first)", ADJ_KEY)
        return 0

    s3 = get_s3_client()

    # Load adjacency edge list
    adj = _load_parquet(s3, ADJ_KEY)
    adj["zcta_id_1"] = adj["zcta_id_1"].astype(str)
    adj["zcta_id_2"] = adj["zcta_id_2"].astype(str)
    log.info("Adjacency: %d edges, columns: %s", len(adj), list(adj.columns))

    # Compute degree per ZCTA
    deg1 = adj.groupby("zcta_id_1").size().rename("deg")
    deg2 = adj.groupby("zcta_id_2").size().rename("deg")
    degree = (deg1.add(deg2, fill_value=0)).astype(int).rename("degree")
    degree.index.name = "zcta_id"
    deg_map = degree.to_dict()
    log.info("Degree: min=%d, median=%d, max=%d",
             degree.min(), int(degree.median()), degree.max())

    adj["degree_1"] = adj["zcta_id_1"].map(deg_map).astype(int)
    adj["degree_2"] = adj["zcta_id_2"].map(deg_map).astype(int)

    # Load boundaries and compute centroids
    bounds = _load_parquet(s3, BOUNDS_KEY)
    centroids = compute_centroids(bounds)
    cent_map = centroids.set_index("zcta_id")[["centroid_lat", "centroid_lon"]]

    # Join centroids to edge list and compute distance
    adj = adj.merge(
        cent_map.rename(columns={"centroid_lat": "lat_1", "centroid_lon": "lon_1"}),
        left_on="zcta_id_1", right_index=True, how="left",
    )
    adj = adj.merge(
        cent_map.rename(columns={"centroid_lat": "lat_2", "centroid_lon": "lon_2"}),
        left_on="zcta_id_2", right_index=True, how="left",
    )

    adj["centroid_distance_km"] = _haversine_km(
        adj["lat_1"].values, adj["lon_1"].values,
        adj["lat_2"].values, adj["lon_2"].values,
    ).round(3)

    # Drop intermediate lat/lon columns
    adj = adj.drop(columns=["lat_1", "lon_1", "lat_2", "lon_2"])

    log.info("Distance: min=%.2f km, median=%.2f km, max=%.2f km",
             adj["centroid_distance_km"].min(),
             adj["centroid_distance_km"].median(),
             adj["centroid_distance_km"].max())

    # Final schema: zcta_id_1, zcta_id_2, degree_1, degree_2, centroid_distance_km
    log.info("Enriched adjacency: %d edges, columns: %s", len(adj), list(adj.columns))

    # Save locally
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / "zcta_adjacency_enriched.parquet"
    adj.to_parquet(local_file, index=False)
    log.info("Written to %s", local_file)

    if args.upload:
        # Archive original
        archive_key = ADJ_KEY.replace(".parquet", f"_pre_enrich_{datetime.now(timezone.utc).strftime('%Y%m%d')}.parquet")
        log.info("Archiving original to %s", archive_key)
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": ADJ_KEY},
            Key=archive_key,
        )

        # Upload enriched
        buf = io.BytesIO()
        adj.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=ADJ_KEY, Body=buf.getvalue())
        log.info("Uploaded enriched adjacency to s3://%s/%s", BUCKET, ADJ_KEY)

        # Upload provenance record
        provenance = {
            "action": "enrich_adjacency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_key": archive_key,
            "enriched_key": ADJ_KEY,
            "columns_added": ["degree_1", "degree_2", "centroid_distance_km"],
            "n_edges": len(adj),
            "n_unique_zctas": len(degree),
            "degree_stats": {
                "min": int(degree.min()),
                "median": int(degree.median()),
                "max": int(degree.max()),
            },
            "distance_stats_km": {
                "min": round(float(adj["centroid_distance_km"].min()), 3),
                "median": round(float(adj["centroid_distance_km"].median()), 3),
                "max": round(float(adj["centroid_distance_km"].max()), 3),
            },
        }
        prov_key = "results/s035/enrich_adjacency_provenance.json"
        upload_json_result(s3, BUCKET, prov_key, provenance)
        log.info("Provenance: s3://%s/%s", BUCKET, prov_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
