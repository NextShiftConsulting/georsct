"""
render_fig5_lisa_clusters.py -- DOE Spatial Diagnostics, Figure 5

Three-panel residual-cluster choropleth (R0 | R1 | R2) for the primary cell
(houston, obs_nfip_event_claims, histgbdt). Each ZCTA colored by LISA cluster
class; panel subtitles carry global Moran's I and the significant-cluster
fraction. Appendix mode grids the same triptych across all modelable cells.

Reads the sidecar parquets produced by compute_spatial_sidecar.py:
    results/s035/sidecar/lisa_{scenario}_{target}_{level}.parquet
        columns: zcta_id, residual, local_moran_I, local_moran_p_raw,
                 local_moran_p_fdr, cluster_label
    results/s035/sidecar/lisa_rollup.parquet
        columns: scenario, target, level, frac_HH, frac_LL, frac_outlier,
                 frac_significant, global_moran_I, n_zctas

Geometry from raw/geocertdb2026/zcta_boundaries_5070.parquet (EPSG:5070).

Conventions matched to existing paper figures: EPSG:5070 equal-area, vector
output (PDF primary for LaTeX, SVG secondary per DOE output spec), GeoDa/splot
canonical LISA colorway so the scheme reads as standard spatial-stats vocabulary.

Usage:
    python render_fig5_lisa_clusters.py --mode main   --out fig5_lisa_clusters
    python render_fig5_lisa_clusters.py --mode appendix --out figA_lisa_grid
    python render_fig5_lisa_clusters.py --mode main --local-dir ./sidecar_pull
"""

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# GeoDa / splot canonical LISA colorway -- reviewers expect this exact scheme.
LISA_COLORS = {
    "HH": "#d7191c",   # high residual surrounded by high -- systematic under-predict cluster
    "LL": "#2c7bb6",   # low surrounded by low
    "HL": "#fdae61",   # high outlier among low neighbors
    "LH": "#abd9e9",   # low outlier among high neighbors
    "ns": "#d9d9d9",   # not significant (FDR)
}
LISA_ORDER = ["HH", "LL", "HL", "LH", "ns"]
LISA_LABELS = {
    "HH": "High-High", "LL": "Low-Low",
    "HL": "High-Low (outlier)", "LH": "Low-High (outlier)",
    "ns": "Not significant",
}

PRIMARY_SCENARIO = "houston"
PRIMARY_TARGET = "obs_nfip_event_claims"
LEVELS = ["r0", "r1", "r2"]
LEVEL_TITLES = {"r0": "R0 (tabular)", "r1": "R1 (+spatial)", "r2": "R2 (+temporal)"}

SIDECAR_PREFIX = "results/s035/sidecar"
BOUNDARY_KEYS = [
    "raw/geocertdb2026/zcta_boundaries_5070.parquet",
    "raw/geocertdb2026/zcta_boundaries.parquet",
]
DPI = 300


# ──────────────────────────────────────────────────────────────────────────
# Loading (S3 by default; --local-dir reads parquets straight off disk)
# ──────────────────────────────────────────────────────────────────────────

def _load_geometry(s3, local_dir):
    import geopandas as gpd

    if local_dir:
        for name in ["zcta_boundaries_5070.parquet", "zcta_boundaries.parquet"]:
            p = Path(local_dir) / name
            if p.exists():
                gdf = gpd.read_parquet(p)
                return _ensure_5070(gdf)
        raise FileNotFoundError(f"No ZCTA boundary parquet under {local_dir}")

    for key in BOUNDARY_KEYS:
        try:
            from _coverage_common import BUCKET
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
            return _ensure_5070(gdf)
        except Exception:
            continue
    raise FileNotFoundError("ZCTA boundaries not found on S3")


def _ensure_5070(gdf):
    if "zcta_id" in gdf.columns:
        gdf["zcta_id"] = gdf["zcta_id"].astype(str)
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        gdf = gdf.to_crs("EPSG:5070")
    return gdf[["zcta_id", "geometry"]]


def _load_lisa(s3, local_dir, scenario, target, level):
    name = f"lisa_{scenario}_{target}_{level}.parquet"
    if local_dir:
        p = Path(local_dir) / name
        if not p.exists():
            return None
        return pd.read_parquet(p).assign(zcta_id=lambda d: d["zcta_id"].astype(str))
    try:
        from _coverage_common import BUCKET
        obj = s3.get_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/{name}")
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        df["zcta_id"] = df["zcta_id"].astype(str)
        return df
    except Exception:
        return None


def _load_rollup(s3, local_dir):
    name = "lisa_rollup.parquet"
    if local_dir:
        p = Path(local_dir) / name
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    try:
        from _coverage_common import BUCKET
        obj = s3.get_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/{name}")
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────

def _subtitle(rollup, scenario, target, level):
    """Global Moran's I + significant-cluster fraction for the panel subtitle."""
    if rollup.empty:
        return ""
    row = rollup[(rollup["scenario"] == scenario)
                 & (rollup["target"] == target)
                 & (rollup["level"] == level)]
    if row.empty:
        return ""
    r = row.iloc[0]
    sig = r.get("frac_HH", 0.0) + r.get("frac_LL", 0.0)
    return f"Moran's I = {r['global_moran_I']:.3f}   |   sig. clusters = {sig:.1%}"


def _draw_panel(ax, gdf, lisa_df, title, subtitle):
    merged = gdf.merge(lisa_df[["zcta_id", "cluster_label"]], on="zcta_id", how="left")
    merged["cluster_label"] = merged["cluster_label"].fillna("ns")
    for label in LISA_ORDER:
        sel = merged[merged["cluster_label"] == label]
        if len(sel):
            sel.plot(ax=ax, color=LISA_COLORS[label], edgecolor="white",
                     linewidth=0.15)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=2)
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#444444")
    ax.set_axis_off()
    ax.set_aspect("equal")


def _legend(fig):
    handles = [mpatches.Patch(color=LISA_COLORS[k], label=LISA_LABELS[k])
               for k in LISA_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))


def render_main(s3, local_dir, out_stem):
    gdf = _load_geometry(s3, local_dir)
    rollup = _load_rollup(s3, local_dir)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    any_panel = False
    for ax, level in zip(axes, LEVELS):
        lisa_df = _load_lisa(s3, local_dir, PRIMARY_SCENARIO, PRIMARY_TARGET, level)
        if lisa_df is None:
            ax.text(0.5, 0.5, f"{level.upper()}\n(no data yet)", ha="center",
                    va="center", transform=ax.transAxes, color="#999999")
            ax.set_axis_off()
            continue
        any_panel = True
        sub = _subtitle(rollup, PRIMARY_SCENARIO, PRIMARY_TARGET, level)
        _draw_panel(ax, gdf, lisa_df, LEVEL_TITLES[level], sub)

    _legend(fig)
    fig.suptitle(
        "Local Moran's I on out-of-fold residuals  |  Houston, NFIP event claims (HistGBDT)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_stem)
    if not any_panel:
        print("WARNING: no LISA parquets found -- rendered placeholder panels only.")


def render_appendix(s3, local_dir, out_stem):
    """Grid: rows = (scenario, target) cells, cols = R0/R1/R2."""
    gdf = _load_geometry(s3, local_dir)
    rollup = _load_rollup(s3, local_dir)
    if rollup.empty:
        print("WARNING: lisa_rollup.parquet not found -- cannot enumerate cells.")
        return
    cells = (rollup[["scenario", "target"]].drop_duplicates()
             .sort_values(["scenario", "target"]).values.tolist())

    n = len(cells)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.4 * n), squeeze=False)
    for i, (scenario, target) in enumerate(cells):
        for j, level in enumerate(LEVELS):
            ax = axes[i][j]
            lisa_df = _load_lisa(s3, local_dir, scenario, target, level)
            if lisa_df is None:
                ax.set_axis_off()
                continue
            title = f"{scenario} / {target}" if j == 0 else LEVEL_TITLES[level]
            _draw_panel(ax, gdf, lisa_df, title, _subtitle(rollup, scenario, target, level))
    _legend(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_stem)


def _save(fig, out_stem):
    pdf_path = f"{out_stem}.pdf"
    svg_path = f"{out_stem}.svg"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=DPI)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path} and {svg_path}")


def main():
    ap = argparse.ArgumentParser(description="Render Figure 5 (LISA residual clusters)")
    ap.add_argument("--mode", choices=["main", "appendix"], default="main")
    ap.add_argument("--out", default="fig5_lisa_clusters")
    ap.add_argument("--local-dir", default=None,
                    help="Read sidecar parquets from a local directory instead of S3")
    args = ap.parse_args()

    s3 = None
    if not args.local_dir:
        from _coverage_common import get_s3_client
        s3 = get_s3_client()

    if args.mode == "main":
        render_main(s3, args.local_dir, args.out)
    else:
        render_appendix(s3, args.local_dir, args.out)


if __name__ == "__main__":
    main()
