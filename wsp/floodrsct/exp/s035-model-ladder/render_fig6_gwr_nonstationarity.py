"""
render_fig6_gwr_nonstationarity.py -- DOE Spatial Diagnostics, Figure 6

Local-R^2 choropleth for the primary cell (houston), showing where the global
feature -> NFIP-claims relationship holds vs collapses, paired with a
small-multiple of local coefficient surfaces.

Reads sidecar outputs from compute_spatial_sidecar.py:
    results/s035/sidecar/gwr_local_r2_houston.parquet   (zcta_id, local_r2)  -- EXISTS
    results/s035/sidecar/gwr_nonstationarity.json       (cell summaries)     -- EXISTS

Geometry from raw/geocertdb2026/zcta_boundaries_5070.parquet (EPSG:5070).

────────────────────────────────────────────────────────────────────────────
KNOWN GAP -- coefficient surfaces are NOT persisted by the sidecar yet.
────────────────────────────────────────────────────────────────────────────
fit_gwr_probe() computes per-ZCTA local coefficients as gwr_result.params but
only keeps scalar summaries (std/|mean| per feature) and discards the surfaces.
The local-R^2 panel renders from existing data; the coefficient small-multiple
needs a one-time sidecar amendment. In run_section_gwr(), alongside the
existing local_r2 upload, add:

    # persist per-ZCTA local coefficients for figure rendering
    params = result_full.pop("local_params", None)   # shape (n_zcta, n_feat)
    if upload and params is not None and zcta_ids:
        pdf = pd.DataFrame(params, columns=available)
        pdf.insert(0, "zcta_id", zcta_ids)
        buf = io.BytesIO(); pdf.to_parquet(buf, compression="zstd")
        s3.put_object(Bucket=BUCKET,
            Key=f"{SIDECAR_PREFIX}/gwr_params_{scenario}.parquet",
            Body=buf.getvalue())

and have fit_gwr_probe() return "local_params": gwr_result.params.tolist()
(it is already computed at line ~350 for the coef-spread index -- just keep it).

Until that lands, this script renders the local-R^2 panel only and prints the
amendment. With gwr_params_houston.parquet present, it renders the full figure.

Usage:
    python render_fig6_gwr_nonstationarity.py --out fig6_gwr_nonstationarity
    python render_fig6_gwr_nonstationarity.py --out fig6 --local-dir ./sidecar_pull
"""

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PRIMARY_SCENARIO = "houston"
SIDECAR_PREFIX = "results/s035/sidecar"
BOUNDARY_KEYS = [
    "raw/geocertdb2026/zcta_boundaries_5070.parquet",
    "raw/geocertdb2026/zcta_boundaries.parquet",
]
# Up to three coefficient surfaces in the small-multiple, in this order if present.
COEF_PRIORITY = ["flood_pct_zone_a", "twi_acc_twi", "svi_overall",
                 "slope_basin_slope", "acs_median_hh_income", "population"]
DPI = 300


def _ensure_5070(gdf):
    if "zcta_id" in gdf.columns:
        gdf["zcta_id"] = gdf["zcta_id"].astype(str)
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        gdf = gdf.to_crs("EPSG:5070")
    return gdf[["zcta_id", "geometry"]]


def _load_geometry(s3, local_dir):
    import geopandas as gpd
    if local_dir:
        for name in ["zcta_boundaries_5070.parquet", "zcta_boundaries.parquet"]:
            p = Path(local_dir) / name
            if p.exists():
                return _ensure_5070(gpd.read_parquet(p))
        raise FileNotFoundError(f"No ZCTA boundary parquet under {local_dir}")
    for key in BOUNDARY_KEYS:
        try:
            from _coverage_common import BUCKET
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            return _ensure_5070(gpd.read_parquet(io.BytesIO(obj["Body"].read())))
        except Exception:
            continue
    raise FileNotFoundError("ZCTA boundaries not found on S3")


def _load_parquet(s3, local_dir, name):
    if local_dir:
        p = Path(local_dir) / name
        if not p.exists():
            return None
        df = pd.read_parquet(p)
    else:
        try:
            from _coverage_common import BUCKET
            obj = s3.get_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/{name}")
            df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception:
            return None
    if "zcta_id" in df.columns:
        df["zcta_id"] = df["zcta_id"].astype(str)
    return df


def _load_gwr_json(s3, local_dir):
    name = "gwr_nonstationarity.json"
    if local_dir:
        p = Path(local_dir) / name
        return json.loads(p.read_text()) if p.exists() else {}
    try:
        from _coverage_common import BUCKET
        obj = s3.get_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/{name}")
        return json.loads(obj["Body"].read())
    except Exception:
        return {}


def _gwr_summary(gwr_json):
    """Pull the houston cell's ΔAICc and local-R^2 range for the caption strip."""
    for c in gwr_json.get("cells", []):
        if c.get("scenario") == PRIMARY_SCENARIO and c.get("status") == "COMPLETE":
            return c
    return {}


def _draw_choropleth(ax, gdf, df, value_col, title, cmap, vmin=None, vmax=None):
    merged = gdf.merge(df[["zcta_id", value_col]], on="zcta_id", how="left")
    merged.plot(ax=ax, column=value_col, cmap=cmap, vmin=vmin, vmax=vmax,
                edgecolor="white", linewidth=0.15, legend=True,
                missing_kwds={"color": "#eeeeee"},
                legend_kwds={"shrink": 0.6})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()
    ax.set_aspect("equal")


def render(s3, local_dir, out_stem):
    gdf = _load_geometry(s3, local_dir)
    lr2 = _load_parquet(s3, local_dir, f"gwr_local_r2_{PRIMARY_SCENARIO}.parquet")
    params = _load_parquet(s3, local_dir, f"gwr_params_{PRIMARY_SCENARIO}.parquet")
    summary = _gwr_summary(_load_gwr_json(s3, local_dir))

    if lr2 is None:
        print("ERROR: gwr_local_r2_houston.parquet not found. Run the GWR section first.")
        return

    coef_cols = []
    if params is not None:
        coef_cols = [c for c in COEF_PRIORITY if c in params.columns][:3]

    n_panels = 1 + len(coef_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.6), squeeze=False)
    axes = axes[0]

    _draw_choropleth(axes[0], gdf, lr2, "local_r2", "Local $R^2$", "viridis",
                     vmin=0, vmax=1)
    if summary:
        strip = (f"$\\Delta$AICc = {summary.get('gwr_delta_aicc', float('nan')):.1f}   "
                 f"local $R^2$ {min(lr2['local_r2']):.2f}–{max(lr2['local_r2']):.2f}")
        axes[0].text(0.5, -0.04, strip, transform=axes[0].transAxes,
                     ha="center", va="top", fontsize=8, color="#444444")

    for ax, col in zip(axes[1:], coef_cols):
        _draw_choropleth(ax, gdf, params, col, f"local $\\beta$: {col}", "RdBu_r")

    if not coef_cols:
        fig.suptitle("GWR non-stationarity (Houston) — local-$R^2$ only; "
                     "coefficient surfaces pending sidecar amendment", fontsize=11, y=1.02)
        print("NOTE: gwr_params_houston.parquet absent. Rendered local-R^2 panel only.")
        print("      See the module docstring for the one-time sidecar amendment.")
    else:
        fig.suptitle("Geographically Weighted Regression — Houston, NFIP event claims",
                     fontsize=12, y=1.02)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    pdf_path, svg_path = f"{out_stem}.pdf", f"{out_stem}.svg"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=DPI)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path} and {svg_path}")


def main():
    ap = argparse.ArgumentParser(description="Render Figure 6 (GWR non-stationarity)")
    ap.add_argument("--out", default="fig6_gwr_nonstationarity")
    ap.add_argument("--local-dir", default=None)
    args = ap.parse_args()
    s3 = None
    if not args.local_dir:
        from _coverage_common import get_s3_client
        s3 = get_s3_client()
    render(s3, args.local_dir, args.out)


if __name__ == "__main__":
    main()
