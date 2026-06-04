#!/usr/bin/env python3
"""render_oahu_summary_table.py -- SIGSPATIAL Appendix: Oahu per-ZCTA summary table.

Reads existing kappa_spatial probe output (residuals_by_zcta.csv, evidence JSON)
and Floodcaster building results, then produces:
  1. LaTeX table fragment (oahu_summary_table.tex)
  2. Summary JSON (oahu_summary.json) with aggregate statistics

The table shows per-ZCTA: buildings (total/flooded), predicted loss, NFIP claims,
normalized residual, median flood depth, and mean structural damage.

Inputs (S3):
  swarm-yrsn-datasets/geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv
  swarm-yrsn-datasets/geocert-experiments/s036/floodcaster_spatial/evidence_s036_h4.json
  swarm-floodcaster/results/1f3ba5fedaaa.parquet
  swarm-floodrsct-data/raw/hawaii/hawaii_zcta_centroids.parquet

Outputs:
  oahu_summary_table.tex
  oahu_summary.json
  -> s3://swarm-floodrsct-data/results/s035/figures/

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Lightweight.

Usage:
    python render_oahu_summary_table.py --upload
    python render_oahu_summary_table.py --local-dir ./outputs
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("render_oahu_summary_table")

# ── Constants ────────────────────────────────────────────────────────────
FLOODCASTER_BUCKET = "swarm-floodcaster"
FLOODRSCT_BUCKET = "swarm-floodrsct-data"
DATASETS_BUCKET = "swarm-yrsn-datasets"
OUTPUT_BUCKET = "swarm-floodrsct-data"
OUTPUT_PREFIX = "results/s035/figures"

OAHU_JOB_ID = "1f3ba5fedaaa"
KAPPA_PREFIX = "geocert-experiments/s036/floodcaster_spatial"
HAWAII_PREFIX = "raw/hawaii"


# ── Data loading ─────────────────────────────────────────────────────────

def load_residuals_csv(s3) -> pd.DataFrame:
    """Load pre-computed per-ZCTA residuals from kappa_spatial probe."""
    key = f"{KAPPA_PREFIX}/residuals_by_zcta.csv"
    log.info("Loading s3://%s/%s", DATASETS_BUCKET, key)
    resp = s3.get_object(Bucket=DATASETS_BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(resp["Body"].read()))
    df["zcta"] = df["zcta"].astype(str)
    log.info("Residuals: %d ZCTAs", len(df))
    return df


def load_evidence_json(s3) -> dict:
    """Load kappa_spatial evidence JSON."""
    key = f"{KAPPA_PREFIX}/evidence_s036_h4.json"
    log.info("Loading s3://%s/%s", DATASETS_BUCKET, key)
    resp = s3.get_object(Bucket=DATASETS_BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode())


def load_buildings(s3) -> pd.DataFrame:
    """Load Floodcaster Oahu building results."""
    key = f"results/{OAHU_JOB_ID}.parquet"
    log.info("Loading s3://%s/%s", FLOODCASTER_BUCKET, key)
    resp = s3.get_object(Bucket=FLOODCASTER_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def load_zcta_centroids(s3) -> pd.DataFrame:
    """Load Hawaii ZCTA centroids for nearest-neighbor assignment."""
    key = f"{HAWAII_PREFIX}/hawaii_zcta_centroids.parquet"
    log.info("Loading s3://%s/%s", FLOODRSCT_BUCKET, key)
    resp = s3.get_object(Bucket=FLOODRSCT_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def assign_zctas(buildings: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Assign buildings to nearest ZCTA centroid (haversine)."""
    lat_col = next(c for c in centroids.columns if "lat" in c.lower())
    lon_col = next(c for c in centroids.columns if "lon" in c.lower() or "lng" in c.lower())
    zcta_col = next(c for c in centroids.columns if "zcta" in c.lower() or "geoid" in c.lower())

    zcta_lats = centroids[lat_col].values
    zcta_lons = centroids[lon_col].values
    zcta_ids = centroids[zcta_col].astype(str).values

    bldg_lats = buildings["latitude"].values
    bldg_lons = buildings["longitude"].values

    assignments = []
    batch_size = 5000
    for start in range(0, len(bldg_lats), batch_size):
        end = min(start + batch_size, len(bldg_lats))
        blat = np.radians(bldg_lats[start:end, None])
        blon = np.radians(bldg_lons[start:end, None])
        zlat = np.radians(zcta_lats[None, :])
        zlon = np.radians(zcta_lons[None, :])
        dlat = blat - zlat
        dlon = blon - zlon
        a = np.sin(dlat / 2) ** 2 + np.cos(blat) * np.cos(zlat) * np.sin(dlon / 2) ** 2
        dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        assignments.extend(zcta_ids[np.argmin(dist, axis=1)])

    buildings = buildings.copy()
    buildings["zcta"] = assignments
    log.info("Assigned %d buildings to %d ZCTAs", len(buildings), buildings["zcta"].nunique())
    return buildings


# ── Per-ZCTA building stats ──────────────────────────────────────────────

def compute_building_stats(buildings: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ZCTA building-level statistics."""
    total = buildings.groupby("zcta").agg(
        n_total=("FloodDepth", "count"),
    ).reset_index()

    flooded = buildings[buildings["FloodDepth"] > 0].copy()
    if len(flooded) == 0:
        total["n_flooded"] = 0
        total["median_depth_ft"] = np.nan
        total["mean_bldg_dmg_pct"] = np.nan
        total["pct_res1"] = np.nan
        return total

    flood_stats = flooded.groupby("zcta").agg(
        n_flooded=("FloodDepth", "count"),
        median_depth_ft=("FloodDepth", "median"),
        mean_bldg_dmg_pct=("BldgDmgPct", "mean"),
    ).reset_index()

    # Percent RES1 among flooded
    flooded["is_res1"] = flooded["occupancy_type"] == "RES1"
    res1_pct = flooded.groupby("zcta")["is_res1"].mean().reset_index()
    res1_pct.columns = ["zcta", "pct_res1"]

    flood_stats = flood_stats.merge(res1_pct, on="zcta", how="left")
    merged = total.merge(flood_stats, on="zcta", how="left")
    merged["n_flooded"] = merged["n_flooded"].fillna(0).astype(int)

    return merged


# ── LaTeX rendering ──────────────────────────────────────────────────────

def render_latex_table(
    residuals: pd.DataFrame,
    bldg_stats: pd.DataFrame,
    evidence: dict,
) -> str:
    """Render a LaTeX table fragment for the appendix."""

    # Merge residuals with building stats
    merged = residuals.merge(bldg_stats, on="zcta", how="left")
    merged = merged.sort_values("zcta")

    lines = []
    lines.append(r"% Generated by render_oahu_summary_table.py")
    lines.append(r"% Source: s3://swarm-floodcaster/results/" + OAHU_JOB_ID + ".parquet")
    lines.append(r"% Kappa spatial: s3://swarm-yrsn-datasets/" + KAPPA_PREFIX + "/")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Oahu per-ZCTA Floodcaster damage summary. "
                 r"Predicted losses from 10-year coastal-surge inundation (with reef "
                 r"attenuation) compared against cumulative NFIP historical claims. "
                 r"Moran's $I = " + f"{evidence['morans_i']:.3f}" + r"$, "
                 r"$\kappa_{\mathrm{spatial}} = " + f"{evidence['kappa_spatial']:.3f}" + r"$.}")
    lines.append(r"  \label{tab:oahu_summary}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    ZCTA & Bldgs & Flooded & Pred.\ Loss & NFIP & $|\hat{r}|$ "
                 r"& Depth (ft) & Dmg (\%) \\")
    lines.append(r"    \midrule")

    for _, row in merged.iterrows():
        zcta = row["zcta"]
        n_total = int(row.get("n_total", 0)) if pd.notna(row.get("n_total")) else 0
        n_flooded = int(row.get("n_flooded", 0)) if pd.notna(row.get("n_flooded")) else 0
        pred_loss = row.get("pred_total_loss", 0)
        nfip = row.get("nfip_total", row.get("nfip_count", 0))
        residual = row.get("residual", 0)
        median_depth = row.get("median_depth_ft", np.nan)
        mean_dmg = row.get("mean_bldg_dmg_pct", np.nan)

        # Format pred loss
        if pred_loss >= 1e6:
            loss_str = f"\\${pred_loss / 1e6:.1f}M"
        elif pred_loss >= 1e3:
            loss_str = f"\\${pred_loss / 1e3:.0f}K"
        elif pred_loss > 0:
            loss_str = f"\\${pred_loss:.0f}"
        else:
            loss_str = "---"

        # Format NFIP
        nfip_val = int(nfip) if pd.notna(nfip) else 0
        nfip_str = f"{nfip_val:,}" if nfip_val > 0 else "0"

        # Format depth and damage
        depth_str = f"{median_depth:.2f}" if pd.notna(median_depth) and n_flooded > 0 else "---"
        dmg_str = f"{mean_dmg:.1f}" if pd.notna(mean_dmg) and n_flooded > 0 else "---"

        lines.append(
            f"    {zcta} & {n_total:,} & {n_flooded:,} & {loss_str} "
            f"& {nfip_str} & {residual:.3f} & {depth_str} & {dmg_str} \\\\"
        )

    lines.append(r"    \midrule")

    # Totals row
    tot_total = int(merged["n_total"].sum()) if "n_total" in merged.columns else 0
    tot_flooded = int(merged["n_flooded"].sum()) if "n_flooded" in merged.columns else 0
    tot_pred = merged["pred_total_loss"].sum()
    tot_nfip = merged.get("nfip_total", merged.get("nfip_count", pd.Series([0]))).sum()
    mean_resid = merged["residual"].mean()

    flooded_rows = merged[merged["n_flooded"] > 0]
    if len(flooded_rows) > 0:
        # Weighted median depth and mean damage across flooded ZCTAs
        weights = flooded_rows["n_flooded"].values
        tot_depth = np.average(flooded_rows["median_depth_ft"].fillna(0).values, weights=weights)
        tot_dmg = np.average(flooded_rows["mean_bldg_dmg_pct"].fillna(0).values, weights=weights)
    else:
        tot_depth = 0
        tot_dmg = 0

    lines.append(
        f"    \\textbf{{Total}} & \\textbf{{{tot_total:,}}} & \\textbf{{{tot_flooded:,}}} "
        f"& \\textbf{{\\${tot_pred / 1e6:.1f}M}} & \\textbf{{{int(tot_nfip):,}}} "
        f"& \\textbf{{{mean_resid:.3f}}} & \\textbf{{{tot_depth:.2f}}} "
        f"& \\textbf{{{tot_dmg:.1f}}} \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \vspace{1mm}")
    lines.append(r"  \raggedright\footnotesize")
    lines.append(r"  \textit{Notes:} Pred.\ Loss = Floodcaster predicted building + content loss. "
                 r"NFIP = cumulative historical claim count (not event-matched). "
                 r"$|\hat{r}|$ = absolute normalized residual (pred $-$ NFIP, both scaled to $[0,1]$). "
                 r"Depth = median FloodDepth among inundated buildings. "
                 r"Dmg = mean HAZUS structural damage \%.")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def build_summary_json(
    residuals: pd.DataFrame,
    bldg_stats: pd.DataFrame,
    evidence: dict,
) -> dict:
    """Build a summary JSON for programmatic consumption."""
    merged = residuals.merge(bldg_stats, on="zcta", how="left")
    tot_total = int(merged["n_total"].sum()) if "n_total" in merged.columns else 0
    tot_flooded = int(merged["n_flooded"].sum()) if "n_flooded" in merged.columns else 0
    tot_pred_loss = float(merged["pred_total_loss"].sum())
    tot_nfip = float(merged.get("nfip_total", merged.get("nfip_count", pd.Series([0]))).sum())

    flooded_rows = merged[merged["n_flooded"] > 0]
    n_zctas_with_damage = len(flooded_rows)

    return {
        "source": f"s3://{FLOODCASTER_BUCKET}/results/{OAHU_JOB_ID}.parquet",
        "kappa_source": f"s3://{DATASETS_BUCKET}/{KAPPA_PREFIX}/",
        "job_id": OAHU_JOB_ID,
        "n_zctas": len(merged),
        "n_zctas_with_damage": n_zctas_with_damage,
        "n_buildings_total": tot_total,
        "n_buildings_flooded": tot_flooded,
        "pct_flooded": round(100.0 * tot_flooded / tot_total, 1) if tot_total > 0 else 0,
        "pred_bldg_loss_usd": round(float(merged["pred_bldg_loss"].sum()), 2),
        "pred_content_loss_usd": round(float(merged["pred_content_loss"].sum()), 2),
        "pred_total_loss_usd": round(tot_pred_loss, 2),
        "nfip_claim_count": int(tot_nfip),
        "mean_residual": round(float(merged["residual"].mean()), 4),
        "morans_i": evidence["morans_i"],
        "kappa_spatial": evidence["kappa_spatial"],
        "interpretation": evidence["interpretation"],
        "per_zcta": [
            {
                "zcta": row["zcta"],
                "n_buildings": int(row.get("n_total", 0)),
                "n_flooded": int(row.get("n_flooded", 0)),
                "pred_total_loss": round(float(row.get("pred_total_loss", 0)), 2),
                "nfip_claims": int(row.get("nfip_total", row.get("nfip_count", 0))),
                "residual": round(float(row.get("residual", 0)), 4),
                "median_depth_ft": round(float(row.get("median_depth_ft", 0)), 2)
                    if pd.notna(row.get("median_depth_ft")) else None,
                "mean_bldg_dmg_pct": round(float(row.get("mean_bldg_dmg_pct", 0)), 1)
                    if pd.notna(row.get("mean_bldg_dmg_pct")) else None,
            }
            for _, row in merged.sort_values("zcta").iterrows()
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--local-dir")
    args = parser.parse_args()

    sys.stdout.flush()

    try:
        from swarm_auth import get_aws_credentials
        s3 = boto3.client("s3", **get_aws_credentials())
    except ImportError:
        s3 = boto3.client("s3")

    # Load pre-computed residuals and evidence
    residuals = load_residuals_csv(s3)
    evidence = load_evidence_json(s3)

    # Load building-level data for depth/damage stats
    buildings = load_buildings(s3)
    centroids = load_zcta_centroids(s3)
    buildings = assign_zctas(buildings, centroids)
    bldg_stats = compute_building_stats(buildings)

    log.info("Building stats: %d ZCTAs, %d with flooded buildings",
             len(bldg_stats), (bldg_stats["n_flooded"] > 0).sum())

    # Render LaTeX table
    tex = render_latex_table(residuals, bldg_stats, evidence)
    summary = build_summary_json(residuals, bldg_stats, evidence)

    # Save
    output_dir = Path(args.local_dir) if args.local_dir else Path("/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / "oahu_summary_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    log.info("Saved %s (%d bytes)", tex_path, tex_path.stat().st_size)

    json_path = output_dir / "oahu_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved %s (%d bytes)", json_path, json_path.stat().st_size)

    # Print table for CloudWatch visibility
    print("\n" + tex + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_zcta"}, indent=2))

    # Upload
    if args.upload:
        for local in [tex_path, json_path]:
            key = f"{OUTPUT_PREFIX}/{local.name}"
            ct = "text/plain" if local.suffix == ".tex" else "application/json"
            s3.upload_file(str(local), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": ct})
            log.info("Uploaded s3://%s/%s", OUTPUT_BUCKET, key)

    log.info("Done.")


if __name__ == "__main__":
    main()
