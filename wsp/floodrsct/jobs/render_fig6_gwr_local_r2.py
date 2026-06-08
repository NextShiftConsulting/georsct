#!/usr/bin/env python3
"""render_fig6_gwr_local_r2.py -- SIGSPATIAL Figure 6: GWR local-R2 choropleth.

Single-panel choropleth of local R2 from GWR fit, with inset bar of per-feature
nonstationarity. Primary cell (houston, obs_nfip_event_claims).

The structural claim this figure licenses:
  "GWR fit improves on global OLS (delta-AICc = [...]), and local R2 ranges
   from [...] to [...], indicating the feature->claims relationship is
   non-stationary across space."

This is a structural description of the full cross-section, never predictive
performance. GWR sees all data and makes no generalization claim.

Inputs (S3, sidecar namespace):
  results/s035/sidecar/gwr_local_r2_{scenario}.parquet
  results/s035/sidecar/spatial_sidecar_{scenario}.json  (GWR summary stats)
  raw/geocertdb2026/zcta_boundaries_5070.parquet

Output:
  fig6_gwr_local_r2_{scenario}.pdf / .svg

Usage:
    python render_fig6_gwr_local_r2.py --scenario houston --upload
    python render_fig6_gwr_local_r2.py --scenario houston --local-dir ./outputs
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colorbar as mcolorbar

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual conventions
# ---------------------------------------------------------------------------

SIDECAR_PREFIX = "results/s035/sidecar"

# Local R2 colormap: light-to-dark blue (matches existing R2 heatmap convention)
R2_CMAP = LinearSegmentedColormap.from_list("r2_blue", ["#f7fbff", "#08306b"])

# Nonstationarity bar colors
NONSTAT_COLOR = "#E69F00"  # Okabe-Ito orange (Environmental family)

FIG_WIDTH = 6.5
FIG_HEIGHT = 3.2
DPI_SAVE = 300
DPI_FIG = 150

PRIMARY_TARGET = "obs_nfip_event_claims"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zcta_boundaries(s3) -> "gpd.GeoDataFrame":
    """Load ZCTA boundaries (EPSG:5070)."""
    import geopandas as gpd
    for key in [
        "raw/geocertdb2026/zcta_boundaries_5070.parquet",
        "raw/geocertdb2026/zcta_boundaries.parquet",
        "raw/geocertdb2026/zcta5_boundaries.parquet",
    ]:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
            if gdf.crs is None or gdf.crs.to_epsg() != 5070:
                gdf = gdf.to_crs(epsg=5070)
            gdf["zcta_id"] = gdf["zcta_id"].astype(str)
            return gdf
        except Exception:
            continue
    raise FileNotFoundError("ZCTA boundaries not found on S3")


def load_gwr_local_r2(s3, scenario: str) -> pd.DataFrame:
    """Load per-ZCTA local R2 from GWR fit."""
    key = f"{SIDECAR_PREFIX}/gwr_local_r2_{scenario}.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    df["zcta_id"] = df["zcta_id"].astype(str)
    return df


def load_gwr_summary(s3, scenario: str) -> dict:
    """Load GWR summary stats (AICc, bandwidth, nonstationarity)."""
    key = f"{SIDECAR_PREFIX}/spatial_sidecar_{scenario}.json"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    payload = json.loads(resp["Body"].read().decode())
    # Find the GWR cell for primary target
    for cell in payload.get("gwr_nonstationarity", {}).get("cells", []):
        if cell.get("target") == PRIMARY_TARGET:
            return cell
    # Fallback: return the first cell or the whole payload
    cells = payload.get("gwr_nonstationarity", {}).get("cells", [])
    return cells[0] if cells else payload


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_gwr_panel(
    zcta_gdf: "gpd.GeoDataFrame",
    local_r2: pd.DataFrame,
    gwr_stats: dict,
    scenario: str,
) -> plt.Figure:
    """Render GWR local-R2 choropleth with nonstationarity inset."""
    import geopandas as gpd

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
    })

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_FIG)

    # Layout: map (left 70%) + nonstationarity bars (right 30%)
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 3], wspace=0.08)
    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # --- Map panel ---
    merged = zcta_gdf[["zcta_id", "geometry"]].merge(
        local_r2[["zcta_id", "local_r2"]], on="zcta_id", how="left"
    )
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry")

    r2_min = float(merged["local_r2"].min()) if merged["local_r2"].notna().any() else 0.0
    r2_max = float(merged["local_r2"].max()) if merged["local_r2"].notna().any() else 1.0

    merged_gdf.plot(
        ax=ax_map, column="local_r2", cmap=R2_CMAP,
        vmin=max(0, r2_min - 0.05), vmax=min(1, r2_max + 0.05),
        linewidth=0.1, edgecolor="#666666",
        missing_kwds={"color": "#e0e0e0", "edgecolor": "#999999", "linewidth": 0.1},
        legend=False,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=R2_CMAP,
        norm=plt.Normalize(vmin=max(0, r2_min - 0.05), vmax=min(1, r2_max + 0.05)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Local $R^2$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Stats subtitle
    delta_aicc = gwr_stats.get("gwr_delta_aicc", "[--]")
    mean_r2 = gwr_stats.get("gwr_local_r2_mean", "[--]")
    if isinstance(delta_aicc, (int, float)):
        delta_str = f"$\\Delta$AICc = {delta_aicc:.1f}"
    else:
        delta_str = f"$\\Delta$AICc = {delta_aicc}"
    if isinstance(mean_r2, (int, float)):
        r2_str = f"mean $R^2$ = {mean_r2:.3f}"
    else:
        r2_str = f"mean $R^2$ = {mean_r2}"

    ax_map.set_title(
        f"GWR local $R^2$: {scenario}\n{delta_str}, {r2_str}, "
        f"range [{r2_min:.2f}, {r2_max:.2f}]",
        fontsize=8,
    )
    ax_map.set_axis_off()

    # --- Nonstationarity bar panel ---
    per_feat = gwr_stats.get("per_feature_nonstationarity", {})
    if per_feat:
        features = sorted(per_feat.keys(), key=lambda k: per_feat[k], reverse=True)
        values = [per_feat[f] for f in features]
        # Shorten feature names for display
        short_names = [f.replace("flood_pct_", "").replace("acs_", "")
                        .replace("median_hh_", "").replace("svi_", "svi:")
                       for f in features]

        y_pos = np.arange(len(features))
        ax_bar.barh(y_pos, values, color=NONSTAT_COLOR, edgecolor="#333", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(short_names, fontsize=6)
        ax_bar.set_xlabel("Coef. spread\n(std / |mean|)", fontsize=7)
        ax_bar.set_title("Non-stationarity\nby feature", fontsize=8)
        ax_bar.invert_yaxis()

        # Threshold line
        threshold = gwr_stats.get("nonstat_threshold", 0.5)
        ax_bar.axvline(threshold, color="#888", linestyle="--", linewidth=0.8)
    else:
        ax_bar.text(0.5, 0.5, "[awaiting\ndata]",
                   ha="center", va="center", fontsize=8, color="#888",
                   transform=ax_bar.transAxes)
        ax_bar.set_axis_off()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Figure 6: GWR local-R2 choropleth")
    parser.add_argument("--scenario", default="houston")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--local-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    s3 = get_s3_client()

    log.info("Loading ZCTA boundaries...")
    zcta_gdf = load_zcta_boundaries(s3)

    log.info("Loading GWR local R2...")
    local_r2 = load_gwr_local_r2(s3, args.scenario)

    # Clip to scenario
    zcta_gdf = zcta_gdf[zcta_gdf["zcta_id"].isin(local_r2["zcta_id"])]
    log.info("Scenario ZCTAs: %d", len(zcta_gdf))

    log.info("Loading GWR summary stats...")
    try:
        gwr_stats = load_gwr_summary(s3, args.scenario)
    except Exception as e:
        log.warning("GWR summary not found (%s); rendering without stats", e)
        gwr_stats = {}

    fig = render_gwr_panel(zcta_gdf, local_r2, gwr_stats, args.scenario)

    # Save
    args.local_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig6_gwr_local_r2_{args.scenario}"

    for fmt in ["pdf", "svg"]:
        path = args.local_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight", format=fmt)
        log.info("Saved %s", path)

    if args.upload:
        for fmt in ["pdf", "svg"]:
            path = args.local_dir / f"{stem}.{fmt}"
            s3_key = f"{SIDECAR_PREFIX}/figures/{stem}.{fmt}"
            s3.put_object(
                Bucket=BUCKET, Key=s3_key,
                Body=path.read_bytes(),
                ContentType="application/pdf" if fmt == "pdf" else "image/svg+xml",
            )
            log.info("Uploaded s3://%s/%s", BUCKET, s3_key)

    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()
