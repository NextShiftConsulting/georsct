#!/usr/bin/env python3
"""render_hazus_depth_damage.py -- SIGSPATIAL Appendix: HAZUS depth-damage curves.

Three-panel figure:
  A. Structural damage (%) vs depth for 4 representative occupancy/foundation curves
  B. Content damage (%) vs depth for the same 4 curves
  C. Oahu observed damage by FloodDepth band (n = flooded buildings)

All data read from authoritative sources:
  - HAZUS curves: sphere.data package (flBldgDmgFn.csv, flContDmgFn.csv)
  - Oahu building results: s3://swarm-floodcaster/results/{JOB_ID}.parquet

Outputs:
  hazus_depth_damage_curves.pdf / .svg
  -> s3://swarm-floodrsct-data/results/s035/figures/

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Lightweight render job.

Usage:
    python render_hazus_depth_damage.py --upload
    python render_hazus_depth_damage.py --local-dir ./outputs
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("render_hazus_depth_damage")

# ── Constants ────────────────────────────────────────────────────────────
FLOODCASTER_BUCKET = "swarm-floodcaster"
OUTPUT_BUCKET = "swarm-floodrsct-data"
OUTPUT_PREFIX = "results/s035/figures"

# Oahu coastal surge job (47,983 buildings, Oahu_10_withReef raster)
OAHU_JOB_ID = "1f3ba5fedaaa"

# Depth columns in HAZUS CSV: ft04m..ft10 = -4ft..+10ft
DEPTH_COLS = [
    "ft04m", "ft03m", "ft02m", "ft01m",
    "ft00",
    "ft01", "ft02", "ft03", "ft04", "ft05",
    "ft06", "ft07", "ft08", "ft09", "ft10",
]
DEPTHS_FT = list(range(-4, 11))  # -4, -3, ..., 10

# Representative curves for the figure (matches Oahu building stock)
# RES1 = single-family residential, COM1 = commercial retail
BLDG_CURVES = [
    (129, "RES1 slab, riverine", "#2e86c1", "-"),
    (658, "RES1 slab, coastal", "#e74c3c", "-"),
    (106, "RES1 basement, coastal", "#e67e22", "--"),
    (217, "COM1 all hazards", "#27ae60", "-"),
]
CONT_CURVES = [
    (45, "RES1 riverine", "#2e86c1", "-"),
    (488, "RES1 coastal", "#e74c3c", "-"),
    (30, "RES1 basement", "#e67e22", "--"),
    (90, "COM1", "#27ae60", "-"),
]

# Depth band config for Panel C
DEPTH_BANDS = [
    (0, 1, "0-1 ft"),
    (1, 2, "1-2 ft"),
    (2, 3, "2-3 ft"),
    (3, 5, "3-5 ft"),
    (5, 100, "5+ ft"),
]
BAND_COLORS = ["#a3d5ff", "#74b9ff", "#0984e3", "#2d3436", "#d63031"]


# ── Data loading ─────────────────────────────────────────────────────────

def load_hazus_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load HAZUS depth-damage CSVs from the sphere.data package."""
    import importlib.resources as pkg_resources

    with (
        pkg_resources.files("sphere.data")
        .joinpath("flBldgDmgFn.csv")
        .open("r", encoding="utf-8-sig") as f
    ):
        bdf = pd.read_csv(f).set_index("BldgDmgFnId")

    with (
        pkg_resources.files("sphere.data")
        .joinpath("flContDmgFn.csv")
        .open("r", encoding="utf-8-sig") as f
    ):
        cdf = pd.read_csv(f).set_index("ContDmgFnId")

    log.info("Loaded HAZUS curves: %d building, %d content", len(bdf), len(cdf))
    return bdf, cdf


def load_oahu_results(s3) -> pd.DataFrame:
    """Load Floodcaster Oahu building results from S3."""
    key = f"results/{OAHU_JOB_ID}.parquet"
    log.info("Loading Oahu results: s3://%s/%s", FLOODCASTER_BUCKET, key)
    resp = s3.get_object(Bucket=FLOODCASTER_BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    log.info("Oahu buildings: %d total, %d flooded",
             len(df), (df["FloodDepth"] > 0).sum())
    return df


def get_curve_values(df: pd.DataFrame, fn_id: int) -> np.ndarray:
    """Extract depth-damage values for a single curve ID."""
    return df.loc[fn_id, DEPTH_COLS].values.astype(float)


# ── Rendering ────────────────────────────────────────────────────────────

def render_figure(
    bdf: pd.DataFrame,
    cdf: pd.DataFrame,
    oahu: pd.DataFrame,
) -> plt.Figure:
    """Render the three-panel HAZUS depth-damage figure."""

    fig = plt.figure(figsize=(11.0, 6.0), facecolor="#fafafa")

    # Panel A: structural damage
    ax_a = fig.add_axes([0.07, 0.32, 0.38, 0.58])
    # Panel B: content damage
    ax_b = fig.add_axes([0.55, 0.32, 0.38, 0.58])
    # Panel C: depth band bars (bottom strip)
    ax_c = fig.add_axes([0.07, 0.06, 0.86, 0.18])

    # ── Oahu depth range shading ──
    flooded = oahu[oahu["FloodDepth"] > 0]
    depth_lo = flooded["FloodDepth"].quantile(0.05)
    depth_hi = flooded["FloodDepth"].quantile(0.95)

    for ax in [ax_a, ax_b]:
        ax.axvspan(depth_lo, depth_hi, color="#e8f5e9", alpha=0.5, zorder=0)

    # ── Panel A: structural curves ──
    for fn_id, label, color, ls in BLDG_CURVES:
        vals = get_curve_values(bdf, fn_id)
        ax_a.plot(DEPTHS_FT, vals, color=color, linewidth=2.2,
                  linestyle=ls, label=f"{label} (ID {fn_id})")

    ax_a.set_title("A. Structural Damage (%)", fontsize=11, fontweight="bold",
                    loc="left", pad=8)
    ax_a.set_xlabel("Depth in Structure (ft)", fontsize=10)
    ax_a.set_ylabel("Damage (%)", fontsize=10)
    ax_a.set_xlim(-4, 10)
    ax_a.set_ylim(0, 105)
    ax_a.set_yticks(range(0, 101, 20))
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.text(
        (depth_lo + depth_hi) / 2, 2, "Oahu flood depth range",
        ha="center", fontsize=7, color="#888",
    )

    # ── Panel B: content curves ──
    for fn_id, label, color, ls in CONT_CURVES:
        vals = get_curve_values(cdf, fn_id)
        ax_b.plot(DEPTHS_FT, vals, color=color, linewidth=2.2,
                  linestyle=ls, label=f"{label} (ID {fn_id})")

    ax_b.set_title("B. Content Damage (%)", fontsize=11, fontweight="bold",
                    loc="left", pad=8)
    ax_b.set_xlabel("Depth in Structure (ft)", fontsize=10)
    ax_b.set_xlim(-4, 10)
    ax_b.set_ylim(0, 105)
    ax_b.set_yticks(range(0, 101, 20))
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_b.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_b.grid(True, alpha=0.3)
    ax_b.text(
        (depth_lo + depth_hi) / 2, 2, "Oahu flood depth range",
        ha="center", fontsize=7, color="#888",
    )

    # ── Panel C: Oahu depth band bar chart ──
    n_flooded = len(flooded)
    band_data = []
    for lo, hi, label in DEPTH_BANDS:
        mask = (flooded["FloodDepth"] >= lo) & (flooded["FloodDepth"] < hi)
        band = flooded[mask]
        band_data.append({
            "label": label,
            "count": len(band),
            "avg_dmg": band["BldgDmgPct"].mean() if len(band) > 0 else 0,
        })

    # Width-proportional bars
    total_count = sum(b["count"] for b in band_data)
    x_cursor = 0.0
    for i, bd in enumerate(band_data):
        width = bd["count"] / total_count if total_count > 0 else 0.2
        ax_c.barh(0, width, left=x_cursor, height=0.6,
                  color=BAND_COLORS[i], edgecolor="#2e86c1", linewidth=0.8)
        # Label inside bar if wide enough
        if width > 0.06:
            text_color = "#fff" if i >= 2 else "#333"
            ax_c.text(
                x_cursor + width / 2, 0.08, bd["label"],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=text_color,
            )
            ax_c.text(
                x_cursor + width / 2, -0.12,
                f"{bd['count']} bldgs, {bd['avg_dmg']:.1f}% dmg",
                ha="center", va="center", fontsize=7, color=text_color,
            )
        elif width > 0.02:
            ax_c.text(
                x_cursor + width / 2, 0.08, bd["label"].split()[0],
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="#fff",
            )
        x_cursor += width

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.4, 0.4)
    ax_c.axis("off")

    # Title and summary for Panel C
    median_depth = flooded["FloodDepth"].median()
    mean_dmg = flooded["BldgDmgPct"].mean()
    total_bldg_loss = flooded["BldgLossUSD"].sum()
    total_cont_loss = flooded["ContentLossUSD"].sum()

    ax_c.set_title(
        f"C. Oahu Observed Damage by Depth Band (n = {n_flooded:,} flooded buildings)",
        fontsize=10, fontweight="bold", loc="left", pad=4,
    )
    ax_c.text(
        0.0, -0.35,
        f"Median depth: {median_depth:.2f} ft  |  "
        f"Mean structural damage: {mean_dmg:.1f}%  |  "
        f"Total building loss: ${total_bldg_loss / 1e6:.1f}M  |  "
        f"Total content loss: ${total_cont_loss / 1e6:.1f}M",
        fontsize=7.5, color="#555", transform=ax_c.transData,
    )

    # Footer
    fig.text(
        0.07, 0.005,
        "Source: FEMA HAZUS-MH Technical Manual, USACE-IWR depth-damage functions. "
        "Floodcaster uses sphere-flood v0.3 (flBldgDmgFn.csv, flContDmgFn.csv, flDmgXRef.csv).",
        fontsize=7, color="#666", fontstyle="italic",
    )

    # Supertitle
    fig.text(
        0.5, 0.97,
        "FEMA HAZUS Depth-Damage Functions (USACE-IWR)",
        fontsize=13, fontweight="bold", ha="center", va="top", color="#1a1a2e",
    )
    fig.text(
        0.5, 0.94,
        "Linear interpolation between tabulated depth points. "
        "Curves selected by occupancy type, foundation, and flood hazard class.",
        fontsize=8.5, ha="center", va="top", color="#555",
    )

    return fig


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true",
                        help="Upload PDF+SVG to S3")
    parser.add_argument("--local-dir",
                        help="Save outputs to local directory instead of /tmp")
    parser.add_argument("--out", default="hazus_depth_damage_curves",
                        help="Output filename stem")
    args = parser.parse_args()

    sys.stdout.flush()

    try:
        from swarm_auth import get_aws_credentials
        s3 = boto3.client("s3", **get_aws_credentials())
    except ImportError:
        s3 = boto3.client("s3")

    # Load data
    bdf, cdf = load_hazus_curves()
    oahu = load_oahu_results(s3)

    # Render
    fig = render_figure(bdf, cdf, oahu)

    # Save
    output_dir = Path(args.local_dir) if args.local_dir else Path("/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["pdf", "svg"]:
        out_path = output_dir / f"{args.out}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=300,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info("Saved %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    plt.close(fig)

    # Upload to S3
    if args.upload:
        for fmt in ["pdf", "svg"]:
            local = output_dir / f"{args.out}.{fmt}"
            key = f"{OUTPUT_PREFIX}/{args.out}.{fmt}"
            content_type = "application/pdf" if fmt == "pdf" else "image/svg+xml"
            s3.upload_file(
                str(local), OUTPUT_BUCKET, key,
                ExtraArgs={"ContentType": content_type},
            )
            log.info("Uploaded s3://%s/%s", OUTPUT_BUCKET, key)

    log.info("Done.")


if __name__ == "__main__":
    main()
