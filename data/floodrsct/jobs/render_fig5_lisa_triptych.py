#!/usr/bin/env python3
"""render_fig5_lisa_triptych.py -- SIGSPATIAL Figure 5: LISA residual-cluster triptych.

Three-panel choropleth: R0 -> R1 -> R2, ZCTAs colored by LISA cluster label
(HH/LL/HL/LH/ns). Primary cell (houston, obs_nfip_event_claims, histgbdt).
Each panel captioned with global Moran's I and significant-cluster fraction.

Inputs (S3, sidecar namespace):
  results/s035/sidecar/lisa_{scenario}_{target}_{level}.parquet
  results/s035/sidecar/lisa_rollup.parquet
  raw/geocertdb2026/zcta_boundaries_5070.parquet

Output:
  fig5_lisa_triptych_{scenario}.pdf / .svg

Conventions:
  - EPSG:5070 (equal-area Albers) for display
  - 300 DPI savefig, serif font, top/right spines removed
  - LISA colors: GeoDa standard (HH=dark red, LL=dark blue, outliers lighter)
  - Full-width figure (6.5" x 2.5")

Usage:
    python render_fig5_lisa_triptych.py --scenario houston --upload
    python render_fig5_lisa_triptych.py --scenario houston --local-dir ./outputs
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# LISA cluster colors -- GeoDa standard, colorblind-distinguishable
CLUSTER_COLORS = {
    "HH": "#d7191c",   # Dark red -- hot-spot cluster
    "LL": "#2c7bb6",   # Dark blue -- cold-spot cluster
    "HL": "#fdae61",   # Light orange -- high outlier among low neighbors
    "LH": "#abd9e9",   # Light blue -- low outlier among high neighbors
    "ns": "#e8e8e8",   # Light gray -- not significant
}

LEVELS = ["r0", "r1", "r2"]
LEVEL_LABELS = {"r0": "R0 (static)", "r1": "R1 (+ hydrology)", "r2": "R2 (+ temporal)"}

# Figure sizing (NeurIPS/SIGSPATIAL full-width)
FIG_WIDTH = 6.5   # inches
FIG_HEIGHT = 2.8  # inches
DPI_SAVE = 300
DPI_FIG = 150

# Primary cell for main-text figure
PRIMARY_TARGET = "obs_nfip_event_claims"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zcta_boundaries(s3) -> "gpd.GeoDataFrame":
    """Load ZCTA boundaries (EPSG:5070) from S3."""
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


def load_lisa_parquet(s3, scenario: str, target: str, level: str) -> pd.DataFrame:
    """Load per-ZCTA LISA results from sidecar namespace."""
    key = f"{SIDECAR_PREFIX}/lisa_{scenario}_{target}_{level}.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    # Sidecar LISA parquets may store ZCTA IDs as 'index' column
    if "zcta_id" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "zcta_id"})
    df["zcta_id"] = df["zcta_id"].astype(str)
    return df


def load_lisa_rollup(s3) -> pd.DataFrame:
    """Load LISA rollup table for global Moran's I and cluster fractions."""
    key = f"{SIDECAR_PREFIX}/lisa_rollup.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_triptych(
    zcta_gdf: "gpd.GeoDataFrame",
    lisa_data: dict[str, pd.DataFrame],
    rollup: pd.DataFrame,
    scenario: str,
    target: str,
) -> plt.Figure:
    """Render 3-panel LISA cluster choropleth.

    Args:
        zcta_gdf: ZCTA boundaries (EPSG:5070)
        lisa_data: {level: per-ZCTA LISA DataFrame}
        rollup: LISA rollup table
        scenario: Scenario name
        target: Target column name
    """
    import geopandas as gpd

    # Style setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_FIG)

    for i, level in enumerate(LEVELS):
        ax = axes[i]

        if level not in lisa_data:
            ax.set_title(f"{LEVEL_LABELS.get(level, level)}\n[no data]", fontsize=8)
            ax.set_axis_off()
            continue

        lisa_df = lisa_data[level]

        # Merge ZCTA boundaries with LISA cluster labels
        merged = zcta_gdf[["zcta_id", "geometry"]].merge(
            lisa_df[["zcta_id", "cluster_label"]], on="zcta_id", how="left"
        )
        merged["cluster_label"] = merged["cluster_label"].fillna("ns")
        merged["color"] = merged["cluster_label"].map(CLUSTER_COLORS)

        # Plot
        merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry")
        merged_gdf.plot(
            ax=ax, color=merged["color"].values,
            linewidth=0.1, edgecolor="#444444",
        )

        # Get global stats from rollup
        row = rollup[
            (rollup["scenario"] == scenario) &
            (rollup["target"] == target) &
            (rollup["level"] == level)
        ]
        if len(row) > 0:
            moran_i = row.iloc[0].get("global_moran_I", None)
            frac_sig = row.iloc[0].get("frac_significant", None)
            subtitle_parts = []
            if moran_i is not None:
                subtitle_parts.append(f"I = {moran_i:.3f}")
            if frac_sig is not None:
                subtitle_parts.append(f"{frac_sig:.0%} sig.")
            subtitle = ", ".join(subtitle_parts)
        else:
            subtitle = "[awaiting data]"

        ax.set_title(f"{LEVEL_LABELS.get(level, level)}\n{subtitle}", fontsize=8)
        ax.set_axis_off()

    # Legend
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS["HH"], edgecolor="#333", label="HH (hot spot)"),
        Patch(facecolor=CLUSTER_COLORS["LL"], edgecolor="#333", label="LL (cold spot)"),
        Patch(facecolor=CLUSTER_COLORS["HL"], edgecolor="#333", label="HL (outlier)"),
        Patch(facecolor=CLUSTER_COLORS["LH"], edgecolor="#333", label="LH (outlier)"),
        Patch(facecolor=CLUSTER_COLORS["ns"], edgecolor="#333", label="ns"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=5, fontsize=7, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Residual spatial clusters: {scenario}",
        fontsize=9, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Figure 5: LISA residual-cluster triptych")
    parser.add_argument("--scenario", default="houston")
    parser.add_argument("--target", default=PRIMARY_TARGET)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--local-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    s3 = get_s3_client()

    # Load boundaries
    log.info("Loading ZCTA boundaries...")
    zcta_gdf = load_zcta_boundaries(s3)

    # Filter to scenario ZCTAs (load one LISA file to get the ZCTA set)
    scenario_zctas = set()
    lisa_data: dict[str, pd.DataFrame] = {}

    for level in LEVELS:
        try:
            df = load_lisa_parquet(s3, args.scenario, args.target, level)
            lisa_data[level] = df
            scenario_zctas.update(df["zcta_id"].unique())
        except Exception as e:
            log.warning("LISA data missing for %s/%s/%s: %s",
                       args.scenario, args.target, level, e)

    if not lisa_data:
        log.error("No LISA data found for any level. Run compute_spatial_sidecar.py first.")
        sys.exit(1)

    # Clip boundaries to scenario
    zcta_gdf = zcta_gdf[zcta_gdf["zcta_id"].isin(scenario_zctas)]
    log.info("Scenario ZCTAs: %d", len(zcta_gdf))

    # Load rollup
    try:
        rollup = load_lisa_rollup(s3)
    except Exception:
        log.warning("LISA rollup not found; rendering without stats")
        rollup = pd.DataFrame()

    # Render
    fig = render_triptych(zcta_gdf, lisa_data, rollup, args.scenario, args.target)

    # Save
    args.local_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig5_lisa_triptych_{args.scenario}"

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
