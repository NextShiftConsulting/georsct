#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: nemotron (NVIDIA, via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Fix S3 client sharing (pass not recreate), add shebang/
#       docstring/logging template, add dry-run early exit, fix SCENARIOS list,
#       remove redundant fig.savefig for S3 upload, add new_orleans, add
#       manifest upload, fix f-strings in logger, add --concurrency guard
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""render_zcta_maps.py -- Phase R4.1: ZCTA choropleth map rendering.

Renders one PNG per ZCTA showing FEMA flood zone coloring for the
target ZCTA (yellow highlight) and its Queen-contiguity neighbors.
Output feeds VLM assessment in Phase R4.3.

Usage:
    python render_zcta_maps.py --scenario houston --upload
    python render_zcta_maps.py --scenario houston --dry-run
"""

import argparse
import io
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, OUTPUT_KEYS, get_s3_client  # noqa: E402
from _s3_result import upload_json_result  # noqa: E402

from yrsn.infrastructure.rendering.matplotlib.io import save_figure_png_only  # noqa: E402
from yrsn.infrastructure.rendering.matplotlib.theme import DEFAULT_THEME  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

# Flood zone color mapping (DOE R4 spec)
ZONE_COLORS = {
    "a": "blue",        # 1% annual chance floodplain
    "x": "lightgray",   # minimal flood hazard
    "x500": "lightblue",  # 0.2% annual chance
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str):
    """Load parquet from S3, try GeoDataFrame first."""
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    try:
        return gpd.read_parquet(buf)
    except Exception:
        buf.seek(0)
        return pd.read_parquet(buf)


def _dominant_flood_color(row: pd.Series) -> str:
    """Pick fill color from the highest flood zone percentage."""
    a = row.get("flood_pct_zone_a", 0.0) or 0.0
    x = row.get("flood_pct_zone_x", 0.0) or 0.0
    x500 = row.get("flood_pct_zone_x500", 0.0) or 0.0
    if pd.isna(a):
        a = 0.0
    if pd.isna(x):
        x = 0.0
    if pd.isna(x500):
        x500 = 0.0

    if a >= x and a >= x500:
        return ZONE_COLORS["a"]
    if x >= a and x >= x500:
        return ZONE_COLORS["x"]
    return ZONE_COLORS["x500"]


def _build_adjacency_dict(adj_df: pd.DataFrame) -> dict:
    """Convert edge-list DataFrame to {zcta_id: set(neighbor_ids)}."""
    d = {}
    for _, row in adj_df.iterrows():
        a = str(row["zcta_id"])
        b = str(row["neighbor_id"])
        d.setdefault(a, set()).add(b)
        d.setdefault(b, set()).add(a)
    return d


def render_one_zcta(
    zcta_id: str,
    boundaries: gpd.GeoDataFrame,
    adj_dict: dict,
    event_df: pd.DataFrame,
    scenario: str,
    out_dir: Path,
    upload: bool,
) -> bool:
    """Render and save a single ZCTA map. Returns True on success."""
    if zcta_id not in boundaries["zcta_id"].values:
        log.warning("ZCTA %s not in boundaries, skipping", zcta_id)
        return False

    target_geom = boundaries.loc[
        boundaries["zcta_id"] == zcta_id, "geometry"
    ].iloc[0]
    minx, miny, maxx, maxy = target_geom.bounds
    bx = (maxx - minx) * 0.2
    by = (maxy - miny) * 0.2

    neighbors = adj_dict.get(zcta_id, set())
    plot_ids = {zcta_id} | neighbors
    subset = boundaries[boundaries["zcta_id"].isin(plot_ids)].copy()
    if subset.empty:
        return False

    # Assign colors
    colors = []
    for zid in subset["zcta_id"]:
        if zid == zcta_id:
            colors.append("yellow")
        elif zid in event_df.index:
            colors.append(_dominant_flood_color(event_df.loc[zid]))
        else:
            colors.append("white")

    DEFAULT_THEME.apply()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    subset.plot(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlim(minx - bx, maxx + bx)
    ax.set_ylim(miny - by, maxy + by)
    ax.set_title("ZCTA %s - %s" % (zcta_id, scenario))

    legend_elements = [
        Patch(facecolor="yellow", edgecolor="black", label="Target ZCTA"),
        Patch(facecolor="blue", edgecolor="black", label="Zone A (1% flood)"),
        Patch(facecolor="lightgray", edgecolor="black", label="Zone X (minimal)"),
        Patch(facecolor="lightblue", edgecolor="black", label="Zone X500 (0.2%)"),
        Patch(facecolor="white", edgecolor="black", label="No data"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    # Save local
    local_path = out_dir / f"{zcta_id}.png"
    save_figure_png_only(fig, local_path, dpi=300, close=False)

    # Upload to S3 (create client per-worker; boto3 clients aren't picklable)
    if upload:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        s3_key = f"{RESULTS_PREFIX}/maps/{scenario}/{zcta_id}.png"
        worker_s3 = get_s3_client()
        worker_s3.put_object(
            Bucket=BUCKET, Key=s3_key,
            Body=buf.getvalue(), ContentType="image/png",
        )

    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase R4.1: ZCTA map rendering")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    scenario = args.scenario

    if args.dry_run:
        log.info("DRY RUN: would render ZCTA maps for %s", scenario)
        log.info("Reads: boundaries GeoParquet + adjacency + event features")
        log.info("Writes: %s/maps/%s/{zcta_id}.png", RESULTS_PREFIX, scenario)
        return 0

    s3 = get_s3_client()

    # Load data
    log.info("Loading boundaries GeoParquet...")
    boundaries = _load_parquet(s3, "raw/geocertdb2026/zcta5_boundaries.parquet")
    boundaries["zcta_id"] = boundaries["zcta_id"].astype(str)

    log.info("Loading adjacency...")
    adj_df = _load_parquet(s3, "raw/geocertdb2026/zcta_adjacency.parquet")
    # Adjacency parquet has zcta_id_1 / zcta_id_2 (not zcta_id / neighbor_id)
    adj_df = adj_df.rename(columns={"zcta_id_1": "zcta_id", "zcta_id_2": "neighbor_id"})
    adj_df["zcta_id"] = adj_df["zcta_id"].astype(str)
    adj_df["neighbor_id"] = adj_df["neighbor_id"].astype(str)
    adj_dict = _build_adjacency_dict(adj_df)

    log.info("Loading event features for %s...", scenario)
    event_df = _load_parquet(s3, OUTPUT_KEYS[scenario])
    event_df["zcta_id"] = event_df["zcta_id"].astype(str)
    # Deduplicate to one row per ZCTA (flood zone pcts are static across events)
    event_df = event_df.drop_duplicates("zcta_id").set_index("zcta_id")

    out_dir = (
        Path(__file__).parent.parent
        / "exp" / "s035-model-ladder" / "maps" / scenario
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    zcta_ids = list(event_df.index)
    workers = max(1, min(os.cpu_count() or 4, 8))
    log.info("Rendering %d ZCTA maps for %s (%d workers)", len(zcta_ids), scenario, workers)

    success = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                render_one_zcta,
                zcta_id, boundaries, adj_dict, event_df,
                scenario, out_dir, args.upload,
            ): zcta_id
            for zcta_id in zcta_ids
        }
        for future in as_completed(futures):
            completed += 1
            try:
                if future.result():
                    success += 1
            except Exception as exc:
                log.error("Map render failed for %s: %s", futures[future], exc)
            if completed % 25 == 0 or completed == len(zcta_ids):
                log.info("  rendered %d / %d", completed, len(zcta_ids))

    log.info("Rendered %d / %d maps to %s", success, len(zcta_ids), out_dir)

    # Manifest
    manifest = {
        "phase": "R4.1_zcta_maps",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rendered": success,
        "n_total": len(zcta_ids),
    }
    local_manifest = out_dir / f"{scenario}_manifest.json"
    with open(local_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    if args.upload:
        key = f"{RESULTS_PREFIX}/maps/{scenario}_manifest.json"
        upload_json_result(s3, BUCKET, key, manifest)
        log.info("Uploaded manifest to s3://%s/%s", BUCKET, key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
