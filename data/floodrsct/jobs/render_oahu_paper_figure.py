#!/usr/bin/env python3
"""render_oahu_paper_figure.py -- Publication figure replacing oahu_h4_results.

Dark-theme, dashboard-inspired static figure for Appendix B (Oahu H4 probe).
Vector PDF + SVG output. Tile-free (ZCTA polygons on solid dark background).

Uses benchmark vocabulary throughout (kappa_spatial, clustering verdict),
NOT certificate/Applications language.

Layout:
  Left strip (20%):  Metrics panel with canonical numbers
  Center (55%):      ZCTA choropleth + building scatter + reef contour + adjacency
  Right (25%):       Normalized comparison bar chart (pred_norm vs nfip_norm)

Canonical numbers (from appendix.tex):
  - Moran's I = -0.358
  - kappa_spatial = 0.352
  - Buildings: 47,983 / 1,676 inundated (3.5%)
  - ZCTAs: 17 (of 20 HI)
  - Predicted loss: $55M ($33M bldg + $22M content)
  - NFIP claims: 379 (cumulative, not event-matched)

Inputs (S3): same as render_oahu_spatial_panel.py
Outputs:
  fig_oahu_h4_probe_{timestamp}.pdf
  fig_oahu_h4_probe_{timestamp}.svg
  -> s3://swarm-yrsn-datasets/geocert-experiments/s036/oahu_paper_figure/

Resource: ml.m5.xlarge (4 vCPU, 16 GB).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLOODCASTER_BUCKET = "swarm-floodcaster"
FLOODRSCT_BUCKET = "swarm-floodrsct-data"
DATASETS_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_PREFIX = "geocert-experiments/s036/oahu_paper_figure"
OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs/oahu_paper_figure"))
JOB_ID = os.environ.get("JOB_ID", "1f3ba5fedaaa")
HAWAII_ZCTA_PREFIX = "968"
REGION = "us-east-1"

OAHU_BBOX = {
    "lon_min": -158.10, "lon_max": -157.65,
    "lat_min": 21.24, "lat_max": 21.50,
}

# Dark theme colors
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BORDER = "#21262d"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#8b949e"
RED = "#e63946"
GREEN = "#2ea043"
BLUE = "#58a6ff"
CYAN = "#00CED1"
ORANGE = "#d29922"

# ---------------------------------------------------------------------------
# S3 helpers (reused from render_oahu_spatial_panel.py)
# ---------------------------------------------------------------------------

def s3_client():
    try:
        from swarm_auth import get_aws_credentials
        return boto3.client("s3", region_name=REGION, **get_aws_credentials())
    except ImportError:
        return boto3.client("s3", region_name=REGION)


def s3_load_parquet(s3, bucket, key):
    log.info("Loading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def s3_load_csv(s3, bucket, key):
    log.info("Loading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def s3_upload_file(s3, local_path, bucket, key):
    log.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hawaii_boundaries(s3):
    import geopandas as gpd
    from shapely import wkb

    df = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                         "raw/geocertdb2026/zcta5_boundaries.parquet")
    zcta_col = next((c for c in df.columns
                     if "zcta" in c.lower() or "geoid" in c.lower()), None)
    if zcta_col is None:
        raise ValueError(f"No ZCTA ID column in: {list(df.columns)}")

    df[zcta_col] = df[zcta_col].astype(str)
    hawaii = df[df[zcta_col].str.startswith(HAWAII_ZCTA_PREFIX)].copy()
    log.info("Filtered to %d Hawaii ZCTAs", len(hawaii))

    geom_col = next((c for c in hawaii.columns
                     if c.lower() in ("geometry", "geom", "wkb_geometry")), None)
    if geom_col is None:
        raise ValueError(f"No geometry column in: {list(hawaii.columns)}")

    hawaii["geometry"] = hawaii[geom_col].apply(
        lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
    )
    hawaii = hawaii.rename(columns={zcta_col: "zcta_id"})
    return gpd.GeoDataFrame(hawaii, geometry="geometry", crs="EPSG:4326")[["zcta_id", "geometry"]]


def load_reef_contour(s3):
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio not available; skipping reef contour")
        return None

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        s3.download_fileobj(FLOODCASTER_BUCKET, "rasters/Oahu_10_withReef.tif", f)
        f.flush()
        raster_path = f.name

    try:
        from pyproj import Transformer

        with rasterio.open(raster_path) as src:
            log.info("Reef raster CRS=%s, shape=%s, bounds=%s",
                     src.crs, src.shape, src.bounds)
            data = src.read(1)
            nrows, ncols = data.shape
            t = src.transform
            xs_native = t[2] + np.arange(ncols) * t[0]
            ys_native = t[5] + np.arange(nrows) * t[4]

            # Reproject grid coordinates from raster CRS to EPSG:4326
            if src.crs and str(src.crs) != "EPSG:4326":
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                xx, yy = np.meshgrid(xs_native, ys_native)
                lons, lats = transformer.transform(xx, yy)
                log.info("Reprojected to lat/lon: lon=[%.2f, %.2f], lat=[%.2f, %.2f]",
                         lons.min(), lons.max(), lats.min(), lats.max())
                return data, {"lons": lons, "lats": lats, "meshgrid": True}
            else:
                xs = xs_native
                ys = ys_native
                # Crop to Oahu bbox
                col_mask = (xs >= OAHU_BBOX["lon_min"]) & (xs <= OAHU_BBOX["lon_max"])
                row_mask = (ys >= OAHU_BBOX["lat_min"]) & (ys <= OAHU_BBOX["lat_max"])
                if col_mask.sum() >= 2 and row_mask.sum() >= 2:
                    data = data[np.ix_(row_mask, col_mask)]
                    xs = xs[col_mask]
                    ys = ys[row_mask]
                log.info("Reef shape after crop: %s", data.shape)
                return data, {"xs": xs, "ys": ys, "meshgrid": False}
    except Exception as exc:
        log.warning("Failed to read reef raster: %s", exc)
        return None
    finally:
        Path(raster_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_figure(
    zcta_gdf,
    residuals: pd.DataFrame,
    buildings: pd.DataFrame,
    adjacency: pd.DataFrame,
    centroids: pd.DataFrame,
    reef_data,
    output_dir: Path,
) -> None:
    """Render dark-theme dashboard-style publication figure."""

    # Merge residuals onto geometry
    residuals["zcta"] = residuals["zcta"].astype(str)
    zcta_gdf["zcta_id"] = zcta_gdf["zcta_id"].astype(str)
    merged = zcta_gdf.merge(residuals, left_on="zcta_id", right_on="zcta", how="inner")
    log.info("Merged %d ZCTAs with residuals", len(merged))

    # Filter buildings to Oahu bbox (exclude any off-island points)
    buildings = buildings[
        (buildings["longitude"] >= OAHU_BBOX["lon_min"])
        & (buildings["longitude"] <= OAHU_BBOX["lon_max"])
        & (buildings["latitude"] >= OAHU_BBOX["lat_min"])
        & (buildings["latitude"] <= OAHU_BBOX["lat_max"])
    ].copy()
    inundated = buildings[buildings["FloodDepth"].notna() & (buildings["FloodDepth"] > 0)].copy()
    dry = buildings[buildings["FloodDepth"].isna() | (buildings["FloodDepth"] <= 0)].copy()
    log.info("Buildings: %d in bbox, %d inundated, %d dry", len(buildings), len(inundated), len(dry))

    # Centroid dict
    centroids["zcta_id"] = centroids["zcta_id"].astype(str)
    cen_dict = {
        row["zcta_id"]: (row["longitude"], row["latitude"])
        for _, row in centroids.iterrows()
    }

    # ----- Figure setup (dark theme) -----
    fig = plt.figure(figsize=(16, 8), dpi=300, facecolor=BG_DARK)

    # Three panels: metrics strip | map | bar chart
    # Bottom 12% reserved for colorbars + footnote
    ax_metrics = fig.add_axes([0.0, 0.0, 0.18, 1.0], facecolor=BG_DARK)
    ax_map = fig.add_axes([0.19, 0.14, 0.52, 0.80], facecolor=BG_DARK)
    ax_bar = fig.add_axes([0.74, 0.14, 0.24, 0.80], facecolor=BG_CARD)

    # ----- METRICS PANEL (left strip) -----
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis("off")

    # Title
    ax_metrics.text(0.5, 0.96, "Oahu H4", fontsize=16, fontweight="bold",
                    color="#ffffff", ha="center", va="top",
                    fontfamily="sans-serif")
    ax_metrics.text(0.5, 0.925, "Clustering Probe", fontsize=10,
                    color=TEXT_SECONDARY, ha="center", va="top")

    # Metric cards
    metrics = [
        ("kappa_spatial", "0.352", "Spatial agreement", RED, 0.84),
        ("Moran's I", "-0.358", "Residual autocorrelation", ORANGE, 0.72),
        ("Verdict", "FAIL", "Clustering not confirmed", RED, 0.60),
        ("Buildings", "47,983", "1,676 inundated (3.5%)", BLUE, 0.48),
        ("Pred. Loss", "\$55M", "\$33M bldg + \$22M content", RED, 0.36),
        ("NFIP Claims", "379", "Cumulative 1978-2023", GREEN, 0.24),
        ("ZCTAs", "17 of 20", "Oahu (968xx prefix)", TEXT_PRIMARY, 0.12),
    ]

    for label, value, detail, color, y in metrics:
        # Card background
        card = FancyBboxPatch(
            (0.05, y - 0.045), 0.9, 0.095,
            boxstyle="round,pad=0.01",
            facecolor=BG_CARD, edgecolor=BORDER, linewidth=0.5,
            transform=ax_metrics.transAxes, clip_on=False,
        )
        ax_metrics.add_patch(card)
        # Left color accent
        ax_metrics.plot([0.05, 0.05], [y - 0.045, y + 0.05],
                        color=color, linewidth=2.5,
                        transform=ax_metrics.transAxes, clip_on=False)
        # Label
        ax_metrics.text(0.12, y + 0.035, label, fontsize=6.5,
                        color=TEXT_SECONDARY, va="center",
                        fontfamily="sans-serif",
                        transform=ax_metrics.transAxes)
        # Value
        ax_metrics.text(0.12, y + 0.005, value, fontsize=14, fontweight="bold",
                        color=color, va="center",
                        fontfamily="monospace",
                        transform=ax_metrics.transAxes)
        # Detail
        ax_metrics.text(0.12, y - 0.025, detail, fontsize=5.5,
                        color=TEXT_SECONDARY, va="center",
                        fontfamily="sans-serif",
                        transform=ax_metrics.transAxes)

    # ----- MAP PANEL (center) -----
    ax_map.set_facecolor(BG_DARK)

    # Residual choropleth
    vmax = max(abs(merged["residual"].min()), abs(merged["residual"].max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    merged.plot(
        ax=ax_map,
        column="residual",
        cmap="RdBu_r",
        norm=norm,
        edgecolor=BLUE,
        linewidth=0.8,
        alpha=0.7,
        legend=False,
    )

    # Reef contour (clipped to land polygons to avoid offshore fragments)
    if reef_data is not None:
        data, meta = reef_data
        try:
            from shapely.ops import unary_union
            from matplotlib.path import Path as MplPath
            from matplotlib.patches import PathPatch

            # Build clip path from merged ZCTA polygons with buffer
            union_geom = unary_union(merged.geometry).buffer(0.005)
            # Convert shapely to matplotlib clip path
            exterior = union_geom.exterior if hasattr(union_geom, 'exterior') else None
            clip_patch = None
            if exterior is not None:
                verts = list(exterior.coords)
                codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
                clip_patch = PathPatch(MplPath(verts, codes), transform=ax_map.transData)

            if meta.get("meshgrid"):
                cs = ax_map.contour(
                    meta["lons"], meta["lats"], data,
                    levels=[0.01],
                    colors=[CYAN],
                    linewidths=1.2,
                    linestyles="solid",
                    alpha=0.8,
                )
            else:
                cs = ax_map.contour(
                    meta["xs"], meta["ys"], data,
                    levels=[0.01],
                    colors=[CYAN],
                    linewidths=1.2,
                    linestyles="solid",
                    alpha=0.8,
                )
            # Clip contour lines to land area
            if clip_patch is not None:
                for coll in cs.collections:
                    coll.set_clip_path(clip_patch)
        except Exception as exc:
            log.warning("Contour failed: %s", exc)

    # Dry buildings
    if len(dry) > 0:
        ax_map.scatter(
            dry["longitude"], dry["latitude"],
            s=0.1, c="#555555", alpha=0.1, zorder=2, rasterized=True,
        )

    # Inundated buildings
    if len(inundated) > 0:
        sc = ax_map.scatter(
            inundated["longitude"], inundated["latitude"],
            s=2.5, c=inundated["FloodDepth"],
            cmap="YlOrRd", vmin=0, vmax=7,
            alpha=0.85, zorder=3, edgecolors="none", rasterized=True,
        )
        # Flood depth colorbar — placed in bottom strip
        cbar_ax = fig.add_axes([0.19, 0.065, 0.22, 0.025])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Flood depth (ft)", fontsize=8, color=TEXT_PRIMARY,
                        fontweight="bold")
        cbar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
        cbar.outline.set_edgecolor(BORDER)

    # Adjacency edges
    active_zctas = set(merged["zcta_id"])
    edge_lines = []
    for _, row in adjacency.iterrows():
        z1, z2 = str(row["zcta_id_1"]), str(row["zcta_id_2"])
        if z1 in cen_dict and z2 in cen_dict and z1 in active_zctas and z2 in active_zctas:
            edge_lines.append([cen_dict[z1], cen_dict[z2]])
    if edge_lines:
        lc = LineCollection(edge_lines, colors=BLUE, linewidths=0.4,
                            alpha=0.25, linestyles="dashed", zorder=4)
        ax_map.add_collection(lc)

    # ZCTA labels
    for _, row in merged.iterrows():
        zid = row["zcta_id"]
        if zid in cen_dict:
            x, y_coord = cen_dict[zid]
            ax_map.text(
                x, y_coord, zid[-3:], fontsize=5, ha="center", va="center",
                fontweight="bold", color="#ffffff",
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG_DARK)],
                zorder=5,
            )

    # Callout annotations
    callouts = [
        ("96816", "99 NFIP, 0 surge\ninland pluvial", (-157.68, 21.30)),
        ("96819", "95 NFIP, 0 surge\nvalley flooding", (-157.88, 21.46)),
        ("96814", "3 NFIP, \$19.5M pred\ncoastal overpredict", (-157.78, 21.27)),
        ("96850", "0 NFIP, \$15.5M pred\nexposed, no history", (-158.05, 21.32)),
    ]
    for zid, note, text_xy in callouts:
        if zid in cen_dict:
            xy = cen_dict[zid]
            ax_map.annotate(
                note, xy=xy, xytext=text_xy,
                fontsize=5, color=TEXT_PRIMARY,
                arrowprops=dict(arrowstyle="-|>", color=TEXT_SECONDARY,
                                lw=0.5, connectionstyle="arc3,rad=0.15"),
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_CARD,
                          edgecolor=BORDER, alpha=0.92),
                zorder=6,
            )

    # Map formatting
    ax_map.set_xlim(OAHU_BBOX["lon_min"], OAHU_BBOX["lon_max"])
    ax_map.set_ylim(OAHU_BBOX["lat_min"], OAHU_BBOX["lat_max"])
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelsize=6, colors=TEXT_SECONDARY)
    for spine in ax_map.spines.values():
        spine.set_color(BORDER)
    ax_map.set_xlabel("Longitude", fontsize=7, color=TEXT_SECONDARY)
    ax_map.set_ylabel("Latitude", fontsize=7, color=TEXT_SECONDARY)

    # Residual colorbar — placed in bottom strip next to flood depth
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar2_ax = fig.add_axes([0.47, 0.065, 0.22, 0.025])
    cbar2 = fig.colorbar(sm, cax=cbar2_ax, orientation="horizontal")
    cbar2.set_label("Normalized mismatch (pred - NFIP)", fontsize=8,
                    color=TEXT_PRIMARY, fontweight="bold")
    cbar2.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
    cbar2.outline.set_edgecolor(BORDER)

    # Map legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color=CYAN, linewidth=1.5, label="Reef flood extent"),
        plt.Line2D([0], [0], color=BLUE, linewidth=0.6, linestyle="--",
                    alpha=0.5, label=f"Adjacency ({len(edge_lines)} edges)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                    markersize=3, label=f"Inundated ({len(inundated):,})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                    markersize=2, label=f"Dry ({len(dry):,})"),
    ]
    leg = ax_map.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
                        framealpha=0.9, edgecolor=BORDER, facecolor=BG_CARD,
                        labelcolor=TEXT_SECONDARY)
    leg.get_frame().set_linewidth(0.5)

    # ----- BAR CHART (right panel) -----
    ax_bar.set_facecolor(BG_CARD)
    for spine in ax_bar.spines.values():
        spine.set_color(BORDER)

    plot_res = merged[["zcta_id", "pred_norm", "nfip_norm"]].copy()
    plot_res = plot_res.sort_values("pred_norm", ascending=True)
    plot_res["label"] = plot_res["zcta_id"].str[-3:]

    y_pos = np.arange(len(plot_res))
    bar_h = 0.6

    ax_bar.barh(y_pos, plot_res["pred_norm"], height=bar_h,
                color=RED, alpha=0.75, label="Predicted (norm)")
    ax_bar.barh(y_pos, -plot_res["nfip_norm"], height=bar_h,
                color=GREEN, alpha=0.75, label="NFIP claims (norm)")

    ax_bar.axvline(0, color=TEXT_SECONDARY, linewidth=0.3, zorder=1)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_res["label"], fontsize=6, color=TEXT_SECONDARY)
    ax_bar.tick_params(axis="x", labelsize=6, colors=TEXT_SECONDARY)
    ax_bar.set_xlabel("Normalized intensity", fontsize=7, color=TEXT_SECONDARY)
    ax_bar.set_title("Per-ZCTA Comparison",
                     fontsize=9, fontweight="bold", color=TEXT_PRIMARY,
                     loc="left", pad=6)
    leg2 = ax_bar.legend(fontsize=5.5, loc="upper right",
                         framealpha=0.9, edgecolor=BORDER, facecolor=BG_CARD,
                         labelcolor=TEXT_SECONDARY)
    leg2.get_frame().set_linewidth(0.5)

    # Subtitle annotations
    ax_bar.text(0.02, 0.96,
                "Red right = model predicts loss\n"
                "Green left = NFIP history present\n"
                "Mismatch = bars on opposite sides",
                transform=ax_bar.transAxes, fontsize=5,
                color=TEXT_SECONDARY, va="top",
                fontfamily="sans-serif", linespacing=1.4)

    # ----- Footnote -----
    fig.text(
        0.19, 0.025,
        "Negative Moran's I (-0.358): adjacent ZCTAs have anti-correlated residuals. "
        "NFIP reference is cumulative (not event-matched); comparison is illustrative, "
        "not adjudicative.",
        fontsize=6.5, color=TEXT_SECONDARY, style="italic",
    )

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for fmt in ("pdf", "svg"):
        out_path = output_dir / f"fig_oahu_h4_probe_{ts}.{fmt}"
        fig.savefig(out_path, format=fmt, bbox_inches="tight",
                    dpi=300, facecolor=BG_DARK)
        log.info("Saved %s", out_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    zcta_gdf = load_hawaii_boundaries(s3)
    residuals = s3_load_csv(
        s3, DATASETS_BUCKET,
        "geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv",
    )
    buildings = s3_load_parquet(s3, FLOODCASTER_BUCKET,
                                f"results/{JOB_ID}.parquet")
    log.info("Loaded %d buildings", len(buildings))

    adjacency = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                                 "raw/hawaii/hawaii_zcta_adjacency.parquet")
    centroids = s3_load_parquet(s3, FLOODRSCT_BUCKET,
                                 "raw/hawaii/hawaii_zcta_centroids.parquet")
    reef_data = load_reef_contour(s3)

    render_figure(zcta_gdf, residuals, buildings, adjacency, centroids,
                  reef_data, OUTPUT_DIR)

    # Upload
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file():
            s3_upload_file(s3, p, DATASETS_BUCKET,
                           f"{S3_OUTPUT_PREFIX}/{p.name}")
    log.info("Done. Outputs at s3://%s/%s/", DATASETS_BUCKET, S3_OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
