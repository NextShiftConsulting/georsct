#!/usr/bin/env python3
"""
render_oahu_spatial_panel.py -- Publication figure: Oahu H4 spatial setup.

Renders a two-panel vector figure from real geometry data:

  Panel A: ZCTA choropleth colored by absolute residual (pred - NFIP),
           inundated buildings overlaid as scatter, adjacency edges,
           reef raster contour at flood depth = 0.

  Panel B: Residual bar chart (ZCTA-level) with pred vs NFIP breakdown.

Inputs (S3):
  swarm-floodrsct-data/raw/geocertdb2026/zcta5_boundaries.parquet
  swarm-floodrsct-data/raw/hawaii/hawaii_zcta_adjacency.parquet
  swarm-floodrsct-data/raw/hawaii/hawaii_zcta_centroids.parquet
  swarm-floodcaster/results/1f3ba5fedaaa.parquet
  swarm-floodcaster/rasters/Oahu_10_withReef.tif
  swarm-yrsn-datasets/geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv

Outputs:
  outputs/oahu_spatial/fig_oahu_spatial_panel.pdf
  outputs/oahu_spatial/fig_oahu_spatial_panel.svg
  Uploaded to s3://swarm-yrsn-datasets/geocert-experiments/s036/oahu_spatial/

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Boundary parquet is 800 MB but we
filter to 20 Hawaii ZCTAs immediately -- peak memory < 2 GB.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Reconfigure stdout for SageMaker cp1252 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOODCASTER_BUCKET = "swarm-floodcaster"
FLOODRSCT_BUCKET = "swarm-floodrsct-data"
DATASETS_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_PREFIX = "geocert-experiments/s036/oahu_spatial"

OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs/oahu_spatial"))

JOB_ID = os.environ.get("JOB_ID", "1f3ba5fedaaa")
HAWAII_ZCTA_PREFIX = "968"
REGION = "us-east-1"

# Oahu bounding box (lon/lat) for clipping raster + plot extent
OAHU_BBOX = {
    "lon_min": -158.10,
    "lon_max": -157.65,
    "lat_min": 21.24,
    "lat_max": 21.50,
}

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def s3_client() -> boto3.client:
    try:
        from swarm_auth import get_aws_credentials
        return boto3.client("s3", region_name=REGION, **get_aws_credentials())
    except ImportError:
        return boto3.client("s3", region_name=REGION)


def s3_load_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    log.info("Loading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def s3_load_csv(s3, bucket: str, key: str) -> pd.DataFrame:
    log.info("Loading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def s3_upload_file(s3, local_path: Path, bucket: str, key: str) -> None:
    log.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)


def upload_outputs(s3, output_dir: Path, prefix: str) -> None:
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            dest_key = f"{prefix.rstrip('/')}/{p.name}"
            s3_upload_file(s3, p, DATASETS_BUCKET, dest_key)


# ---------------------------------------------------------------------------
# Step 1: Load ZCTA boundaries (filter to Hawaii from national file)
# ---------------------------------------------------------------------------


def load_hawaii_boundaries(s3) -> "gpd.GeoDataFrame":
    import geopandas as gpd
    from shapely import wkb

    log.info("Loading national ZCTA boundaries (filtering to Hawaii 968xx)...")
    df = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                         "raw/geocertdb2026/zcta5_boundaries.parquet")

    # Identify ZCTA ID column
    zcta_col = next((c for c in df.columns
                     if "zcta" in c.lower() or "geoid" in c.lower()), None)
    if zcta_col is None:
        raise ValueError(f"No ZCTA ID column found in: {list(df.columns)}")

    df[zcta_col] = df[zcta_col].astype(str)
    hawaii = df[df[zcta_col].str.startswith(HAWAII_ZCTA_PREFIX)].copy()
    log.info("Filtered to %d Hawaii ZCTAs from %d national", len(hawaii), len(df))

    # Parse geometry
    geom_col = next((c for c in hawaii.columns
                     if c.lower() in ("geometry", "geom", "wkb_geometry")), None)
    if geom_col is None:
        raise ValueError(f"No geometry column found in: {list(hawaii.columns)}")

    hawaii["geometry"] = hawaii[geom_col].apply(
        lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
    )
    hawaii = hawaii.rename(columns={zcta_col: "zcta_id"})

    gdf = gpd.GeoDataFrame(hawaii, geometry="geometry", crs="EPSG:4326")
    return gdf[["zcta_id", "geometry"]]


# ---------------------------------------------------------------------------
# Step 2: Load reef raster and extract flood contour
# ---------------------------------------------------------------------------


def load_reef_contour(s3) -> tuple[np.ndarray, dict] | None:
    """Load Oahu_10_withReef.tif, return (depth_array, transform_meta) or None."""
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio not available; skipping reef contour")
        return None

    log.info("Loading reef raster...")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        s3.download_fileobj(FLOODCASTER_BUCKET,
                            "rasters/Oahu_10_withReef.tif", f)
        f.flush()
        raster_path = f.name

    try:
        with rasterio.open(raster_path) as src:
            # Read within Oahu bbox to limit memory
            from rasterio.windows import from_bounds
            window = from_bounds(
                OAHU_BBOX["lon_min"], OAHU_BBOX["lat_min"],
                OAHU_BBOX["lon_max"], OAHU_BBOX["lat_max"],
                transform=src.transform,
            )
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)

            # Build coordinate arrays for contour plotting
            nrows, ncols = data.shape
            cols = np.arange(ncols)
            rows = np.arange(nrows)
            xs = win_transform[2] + cols * win_transform[0]
            ys = win_transform[5] + rows * win_transform[4]

            return data, {"xs": xs, "ys": ys, "transform": win_transform}
    except Exception as exc:
        log.warning("Failed to read reef raster: %s", exc)
        return None
    finally:
        Path(raster_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 3: Render
# ---------------------------------------------------------------------------


def render_figure(
    zcta_gdf: "gpd.GeoDataFrame",
    residuals: pd.DataFrame,
    buildings: pd.DataFrame,
    adjacency: pd.DataFrame,
    centroids: pd.DataFrame,
    reef_data: tuple | None,
    output_dir: Path,
) -> None:
    """Render two-panel publication figure."""

    # Merge residuals onto geometry
    residuals["zcta"] = residuals["zcta"].astype(str)
    zcta_gdf["zcta_id"] = zcta_gdf["zcta_id"].astype(str)
    merged = zcta_gdf.merge(residuals, left_on="zcta_id", right_on="zcta",
                            how="inner")
    log.info("Merged %d ZCTAs with geometry + residuals", len(merged))

    # Identify inundated buildings
    inundated = buildings[
        buildings["FloodDepth"].notna() & (buildings["FloodDepth"] > 0)
    ].copy()
    dry = buildings[
        buildings["FloodDepth"].isna() | (buildings["FloodDepth"] <= 0)
    ].copy()
    log.info("Buildings: %d total, %d inundated, %d dry",
             len(buildings), len(inundated), len(dry))

    # Centroid dict for adjacency edges
    centroids["zcta_id"] = centroids["zcta_id"].astype(str)
    cen_dict = {
        row["zcta_id"]: (row["longitude"], row["latitude"])
        for _, row in centroids.iterrows()
    }

    # ----- Figure setup -----
    fig = plt.figure(figsize=(14, 7.5), dpi=150)

    # Panel A: spatial map (wider)
    ax_map = fig.add_axes([0.04, 0.08, 0.58, 0.84])
    # Panel B: residual bars (narrower)
    ax_bar = fig.add_axes([0.68, 0.08, 0.30, 0.84])

    # ----- Panel A: Choropleth -----
    # Color by signed residual (pred_norm - nfip_norm)
    vmax = max(abs(merged["residual"].min()), abs(merged["residual"].max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    merged.plot(
        ax=ax_map,
        column="residual",
        cmap="RdBu_r",
        norm=norm,
        edgecolor="#333",
        linewidth=0.6,
        alpha=0.75,
        legend=False,
    )

    # Reef raster contour (flood extent boundary)
    if reef_data is not None:
        data, meta = reef_data
        # Contour at depth = 0 (flood/no-flood boundary)
        try:
            ax_map.contour(
                meta["xs"], meta["ys"], data,
                levels=[0.01],
                colors=["#00CED1"],
                linewidths=0.8,
                linestyles="solid",
                alpha=0.7,
            )
        except Exception as exc:
            log.warning("Contour failed: %s", exc)

    # Dry buildings (very light, small)
    if len(dry) > 0:
        ax_map.scatter(
            dry["longitude"], dry["latitude"],
            s=0.15, c="#cccccc", alpha=0.15, zorder=2, rasterized=True,
        )

    # Inundated buildings colored by flood depth
    if len(inundated) > 0:
        sc = ax_map.scatter(
            inundated["longitude"], inundated["latitude"],
            s=3, c=inundated["FloodDepth"],
            cmap="YlOrRd", vmin=0, vmax=7,
            alpha=0.8, zorder=3, edgecolors="none", rasterized=True,
        )
        # Colorbar for flood depth
        cbar_ax = fig.add_axes([0.04, 0.04, 0.25, 0.02])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Flood depth (ft)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Adjacency edges
    active_zctas = set(merged["zcta_id"])
    edge_lines = []
    for _, row in adjacency.iterrows():
        z1, z2 = str(row["zcta_id_1"]), str(row["zcta_id_2"])
        if z1 in cen_dict and z2 in cen_dict and z1 in active_zctas and z2 in active_zctas:
            edge_lines.append([cen_dict[z1], cen_dict[z2]])
    if edge_lines:
        lc = LineCollection(edge_lines, colors="#555", linewidths=0.5,
                            alpha=0.4, zorder=4)
        ax_map.add_collection(lc)

    # ZCTA labels at centroids
    for _, row in merged.iterrows():
        zid = row["zcta_id"]
        if zid in cen_dict:
            x, y = cen_dict[zid]
            label = zid[-3:]  # last 3 digits
            ax_map.text(
                x, y, label, fontsize=5.5, ha="center", va="center",
                fontweight="bold", color="#111",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                zorder=5,
            )

    # Map formatting
    ax_map.set_xlim(OAHU_BBOX["lon_min"], OAHU_BBOX["lon_max"])
    ax_map.set_ylim(OAHU_BBOX["lat_min"], OAHU_BBOX["lat_max"])
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude", fontsize=9)
    ax_map.set_ylabel("Latitude", fontsize=9)
    ax_map.tick_params(labelsize=7)
    ax_map.set_title(
        "A.  Oahu H4 Spatial Setup: ZCTA Residuals & Inundation",
        fontsize=11, fontweight="bold", loc="left", pad=8,
    )

    # Colorbar for residual choropleth
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar2_ax = fig.add_axes([0.35, 0.04, 0.25, 0.02])
    cbar2 = fig.colorbar(sm, cax=cbar2_ax, orientation="horizontal")
    cbar2.set_label("Residual (pred - NFIP, normalized)", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    # Legend
    legend_elements = [
        Patch(facecolor="#ddd", edgecolor="#333", label="ZCTA boundary"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ccc",
                    markersize=3, label=f"Dry buildings ({len(dry):,})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                    markersize=4, label=f"Inundated ({len(inundated):,})"),
        plt.Line2D([0], [0], color="#555", linewidth=0.8,
                    label=f"Adjacency ({len(edge_lines)} edges)"),
    ]
    if reef_data is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="#00CED1", linewidth=1.2,
                       label="Reef flood extent")
        )
    ax_map.legend(handles=legend_elements, loc="lower left", fontsize=7,
                  framealpha=0.9, edgecolor="#ccc")

    # ----- Panel B: Residual bar chart -----
    # Sort by absolute residual descending
    plot_res = merged[["zcta_id", "pred_norm", "nfip_norm", "residual",
                        "n_buildings", "pred_total_loss", "nfip_total"]].copy()
    plot_res = plot_res.sort_values("residual", key=abs, ascending=True)
    plot_res["label"] = plot_res["zcta_id"].str[-3:]

    y_pos = np.arange(len(plot_res))
    bar_height = 0.65

    # Stacked horizontal bars: pred (blue) and NFIP (orange)
    ax_bar.barh(y_pos, plot_res["pred_norm"], height=bar_height,
                color="#2980b9", alpha=0.8, label="Predicted (norm)")
    ax_bar.barh(y_pos, -plot_res["nfip_norm"], height=bar_height,
                color="#e67e22", alpha=0.8, label="NFIP claims (norm)")

    # Zero line
    ax_bar.axvline(0, color="#333", linewidth=0.5, zorder=1)

    # Labels
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_res["label"], fontsize=7)
    ax_bar.set_xlabel("Normalized value", fontsize=9)
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_title(
        "B.  Per-ZCTA Pred vs NFIP",
        fontsize=11, fontweight="bold", loc="left", pad=8,
    )
    ax_bar.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Annotate key stats in top-right of bar panel
    stats_text = (
        f"Moran's I = -0.358\n"
        f"kappa_spatial = 0.352\n"
        f"n = 17 ZCTAs, 46 edges\n"
        f"Verdict: FAIL\n"
        f"\n"
        f"Buildings: {len(buildings):,}\n"
        f"Inundated: {len(inundated):,} ({100*len(inundated)/len(buildings):.1f}%)\n"
        f"Pred loss: ${merged['pred_total_loss'].sum():,.0f}\n"
        f"NFIP claims: ${merged['nfip_total'].sum():,.0f}"
    )
    ax_bar.text(
        0.97, 0.97, stats_text,
        transform=ax_bar.transAxes, fontsize=7,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9",
                  edgecolor="#ccc", alpha=0.95),
        family="monospace",
    )

    # Interpretation footnote
    fig.text(
        0.04, 0.005,
        "Negative Moran's I: residuals in adjacent ZCTAs are anti-correlated. "
        "Inland ZCTAs (816, 819, 822) have NFIP claims from pluvial/fluvial flooding "
        "but zero predicted loss from coastal-only raster.",
        fontsize=7, style="italic", color="#555",
    )

    # Save
    for fmt in ("pdf", "svg"):
        out_path = output_dir / f"fig_oahu_spatial_panel.{fmt}"
        fig.savefig(out_path, format=fmt, bbox_inches="tight",
                    dpi=300 if fmt == "pdf" else 150)
        log.info("Saved %s", out_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    # 1. ZCTA boundaries (filter national -> Hawaii)
    zcta_gdf = load_hawaii_boundaries(s3)

    # 2. Per-ZCTA residuals (from H4 probe)
    residuals = s3_load_csv(
        s3, DATASETS_BUCKET,
        "geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv",
    )

    # 3. Building locations (from floodcaster output)
    buildings = s3_load_parquet(s3, FLOODCASTER_BUCKET,
                                f"results/{JOB_ID}.parquet")
    log.info("Loaded %d buildings", len(buildings))

    # 4. Adjacency edges
    adjacency = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                                 "raw/hawaii/hawaii_zcta_adjacency.parquet")

    # 5. Centroids
    centroids = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                                 "raw/hawaii/hawaii_zcta_centroids.parquet")

    # 6. Reef raster contour
    reef_data = load_reef_contour(s3)

    # 7. Render
    render_figure(zcta_gdf, residuals, buildings, adjacency, centroids,
                  reef_data, OUTPUT_DIR)

    # 8. Upload
    upload_outputs(s3, OUTPUT_DIR, S3_OUTPUT_PREFIX)
    log.info("Done. Outputs at s3://%s/%s/", DATASETS_BUCKET, S3_OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
