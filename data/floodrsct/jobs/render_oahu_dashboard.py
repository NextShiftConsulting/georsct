#!/usr/bin/env python3
"""render_oahu_dashboard.py -- Interactive Oahu H4 flood risk dashboard.

Produces a self-contained HTML dashboard inspired by the Johns Hopkins
COVID-19 dashboard: dark theme, central interactive map, KPI metric cards,
and comparison charts.

Inputs (S3):
  swarm-floodrsct-data/raw/geocertdb2026/zcta5_boundaries.parquet
  swarm-floodcaster/results/1f3ba5fedaaa.parquet
  swarm-yrsn-datasets/geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv

Outputs:
  outputs/oahu_dashboard/oahu_flood_dashboard_{timestamp}.html
  Uploaded to s3://swarm-yrsn-datasets/geocert-experiments/s036/oahu_dashboard/

Resource: ml.m5.xlarge (4 vCPU, 16 GB).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3():
    """Create S3 client with swarm_auth if available, else bare boto3."""
    try:
        from swarm_auth import get_aws_credentials
        return boto3.client("s3", **get_aws_credentials())
    except ImportError:
        return boto3.client("s3")


def _download(bucket: str, key: str, local: str) -> str:
    log.info("Downloading s3://%s/%s -> %s", bucket, key, local)
    _s3().download_file(bucket, key, local)
    return local


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_buildings(tmp: str) -> pd.DataFrame:
    """Load 47,983 Oahu building results."""
    path = _download(
        "swarm-floodcaster",
        "results/1f3ba5fedaaa.parquet",
        f"{tmp}/buildings.parquet",
    )
    df = pd.read_parquet(path)
    log.info("Buildings loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_residuals(tmp: str) -> pd.DataFrame:
    """Load ZCTA-level residuals (pred vs NFIP)."""
    path = _download(
        "swarm-yrsn-datasets",
        "geocert-experiments/s036/floodcaster_spatial/residuals_by_zcta.csv",
        f"{tmp}/residuals.csv",
    )
    df = pd.read_csv(path)
    log.info("Residuals loaded: %d ZCTAs", len(df))
    return df


def load_zcta_boundaries(tmp: str) -> "gpd.GeoDataFrame":
    """Load Hawaii ZCTA boundaries from national file."""
    import geopandas as gpd

    path = _download(
        "swarm-floodrsct-data",
        "raw/geocertdb2026/zcta5_boundaries.parquet",
        f"{tmp}/zcta5_boundaries.parquet",
    )
    log.info("Reading national ZCTA boundaries (800 MB)...")
    gdf = gpd.read_parquet(path)

    # Auto-detect ZCTA ID column (name varies across census vintages)
    zcta_col = next((c for c in gdf.columns
                     if "zcta" in c.lower() or "geoid" in c.lower()), None)
    if zcta_col is None:
        raise ValueError(f"No ZCTA ID column found in: {list(gdf.columns)}")

    gdf[zcta_col] = gdf[zcta_col].astype(str)
    hawaii = gdf[gdf[zcta_col].str.startswith("968")].copy()
    hawaii = hawaii.rename(columns={zcta_col: "ZCTA5CE20"})
    log.info("Filtered to %d Hawaii ZCTAs (col=%s)", len(hawaii), zcta_col)
    # Ensure WGS84
    if hawaii.crs and hawaii.crs.to_epsg() != 4326:
        hawaii = hawaii.to_crs(epsg=4326)
    return hawaii


# ---------------------------------------------------------------------------
# GeoJSON generation
# ---------------------------------------------------------------------------

def zcta_to_geojson(gdf: "gpd.GeoDataFrame", residuals: pd.DataFrame) -> str:
    """Merge residuals into ZCTA boundaries and export GeoJSON."""
    merged = gdf.merge(
        residuals,
        left_on="ZCTA5CE20",
        right_on="zcta",
        how="inner",
    )
    log.info("Merged ZCTAs with residuals: %d features", len(merged))

    features = []
    for _, row in merged.iterrows():
        geom = row.geometry.__geo_interface__
        props = {
            "zcta": str(row["zcta"]),
            "pred_total_loss": float(row.get("pred_total_loss", 0)),
            "nfip_total": float(row.get("nfip_total", 0)),
            "residual": float(row.get("residual", 0)),
            "n_buildings": int(row.get("n_buildings", 0)),
        }
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    return json.dumps({"type": "FeatureCollection", "features": features})


def buildings_to_geojson(df: pd.DataFrame, sample_n: int = 5000) -> str:
    """Sample buildings and export as GeoJSON points.

    Full 48k points overwhelm browsers. Sample for interactivity,
    but keep the distribution representative by stratified sampling
    on flood depth quantiles.
    """
    if len(df) > sample_n:
        df = df.copy()
        df["_q"] = pd.qcut(df["FloodDepth"].clip(lower=0), q=10, labels=False, duplicates="drop")
        sampled = df.groupby("_q", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_n // 10), random_state=42)
        )
        df = sampled.drop(columns=["_q"])
        log.info("Sampled %d buildings (stratified by flood depth)", len(df))

    features = []
    for _, row in df.iterrows():
        lon = float(row.get("longitude", row.get("lon", row.get("Longitude", 0))))
        lat = float(row.get("latitude", row.get("lat", row.get("Latitude", 0))))
        depth = float(row.get("FloodDepth", 0))
        loss = float(row.get("BldgLossUSD", 0))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "depth_ft": round(depth, 1),
                "loss_usd": round(loss, 0),
            },
        })

    return json.dumps({"type": "FeatureCollection", "features": features})


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(buildings: pd.DataFrame, residuals: pd.DataFrame) -> dict:
    """Compute dashboard headline numbers."""
    total_buildings = len(buildings)
    total_pred_loss = residuals["pred_total_loss"].sum()
    total_nfip = residuals["nfip_total"].sum()
    n_zctas = len(residuals)
    max_depth = buildings["FloodDepth"].max()
    mean_depth = buildings["FloodDepth"].mean()
    pct_inundated = (buildings["FloodDepth"] > 0).mean() * 100

    # Spatial metrics from H4
    kappa_spatial = 0.352
    morans_i = -0.358

    return {
        "total_buildings": total_buildings,
        "total_pred_loss": total_pred_loss,
        "total_nfip": total_nfip,
        "n_zctas": n_zctas,
        "max_depth_ft": float(max_depth),
        "mean_depth_ft": float(mean_depth),
        "pct_inundated": float(pct_inundated),
        "kappa_spatial": kappa_spatial,
        "morans_i": morans_i,
    }


# ---------------------------------------------------------------------------
# HTML dashboard template
# ---------------------------------------------------------------------------

def build_dashboard_html(
    zcta_geojson: str,
    buildings_geojson: str,
    kpis: dict,
    residuals: pd.DataFrame,
    timestamp: str,
) -> str:
    """Build self-contained HTML dashboard."""

    # Prepare bar chart data
    res_sorted = residuals.sort_values("pred_total_loss", ascending=False)
    bar_labels = json.dumps(res_sorted["zcta"].astype(str).tolist())
    bar_pred = json.dumps(res_sorted["pred_total_loss"].tolist())
    bar_nfip = json.dumps(res_sorted["nfip_total"].tolist())

    def fmt_money(v):
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        if v >= 1e6:
            return f"${v/1e6:.1f}M"
        if v >= 1e3:
            return f"${v/1e3:.0f}K"
        return f"${v:.0f}"

    def fmt_num(v):
        return f"{v:,.0f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oahu Flood Risk Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0a0a1a;
    color: #e0e0e0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow-x: hidden;
  }}

  /* --- Header --- */
  .header {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 2px solid #e63946;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}

  .header-title {{
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #ffffff;
  }}

  .header-title span {{
    color: #e63946;
  }}

  .header-subtitle {{
    font-size: 12px;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
  }}

  .header-badge {{
    background: #e63946;
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}

  /* --- Main layout --- */
  .dashboard {{
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    grid-template-rows: auto 1fr;
    height: calc(100vh - 64px);
    gap: 0;
  }}

  /* --- KPI sidebar (left) --- */
  .kpi-panel {{
    background: #0d1117;
    border-right: 1px solid #21262d;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }}

  .kpi-card {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.2s;
  }}

  .kpi-card:hover {{
    border-color: #e63946;
  }}

  .kpi-card.alert {{
    border-left: 3px solid #e63946;
  }}

  .kpi-card.ok {{
    border-left: 3px solid #2ea043;
  }}

  .kpi-card.warn {{
    border-left: 3px solid #d29922;
  }}

  .kpi-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    margin-bottom: 6px;
  }}

  .kpi-value {{
    font-size: 28px;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
    line-height: 1.1;
  }}

  .kpi-value.red {{ color: #e63946; }}
  .kpi-value.orange {{ color: #d29922; }}
  .kpi-value.green {{ color: #2ea043; }}
  .kpi-value.blue {{ color: #58a6ff; }}

  .kpi-detail {{
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* --- Map (center) --- */
  .map-container {{
    position: relative;
  }}

  #map {{
    width: 100%;
    height: 100%;
    background: #0a0a1a;
  }}

  .map-overlay {{
    position: absolute;
    bottom: 24px;
    left: 24px;
    z-index: 1000;
    background: rgba(13, 17, 23, 0.92);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    backdrop-filter: blur(8px);
  }}

  .legend-title {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
    margin-bottom: 8px;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 12px;
  }}

  .legend-swatch {{
    width: 16px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  /* --- Charts panel (right) --- */
  .charts-panel {{
    background: #0d1117;
    border-left: 1px solid #21262d;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }}

  .chart-card {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
  }}

  .chart-title {{
    font-size: 13px;
    font-weight: 700;
    color: #e0e0e0;
    margin-bottom: 12px;
  }}

  .chart-canvas-wrap {{
    position: relative;
    height: 220px;
  }}

  /* --- Verdict banner --- */
  .verdict-banner {{
    background: linear-gradient(90deg, rgba(230, 57, 70, 0.15) 0%, rgba(13, 17, 23, 0) 100%);
    border: 1px solid #e63946;
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .verdict-icon {{
    font-size: 24px;
    line-height: 1;
  }}

  .verdict-text {{
    font-size: 13px;
    color: #e0e0e0;
  }}

  .verdict-text strong {{
    color: #e63946;
    font-weight: 700;
  }}

  /* --- Stats table --- */
  .stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
  }}

  .stats-table th {{
    text-align: left;
    color: #8b949e;
    font-weight: 600;
    padding: 6px 8px;
    border-bottom: 1px solid #21262d;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}

  .stats-table td {{
    padding: 6px 8px;
    border-bottom: 1px solid #161b22;
    color: #e0e0e0;
  }}

  .stats-table tr:hover td {{
    background: rgba(88, 166, 255, 0.05);
  }}

  /* --- Footer --- */
  .footer {{
    grid-column: 1 / -1;
    background: #0d1117;
    border-top: 1px solid #21262d;
    padding: 8px 32px;
    font-size: 11px;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    justify-content: space-between;
  }}

  /* Leaflet dark tiles override */
  .leaflet-tile-pane {{ filter: brightness(0.6) contrast(1.3) saturate(0.3); }}
  .leaflet-container {{ background: #0a0a1a; }}

  /* Custom popup */
  .leaflet-popup-content-wrapper {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
  }}
  .leaflet-popup-tip {{ background: #161b22; }}

  .popup-zcta {{
    font-size: 16px;
    font-weight: 700;
    color: #58a6ff;
    margin-bottom: 8px;
  }}
  .popup-row {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 2px 0;
  }}
  .popup-label {{ color: #8b949e; }}
  .popup-val {{ font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <div class="header-title">Oahu <span>Flood Risk</span> Dashboard</div>
    <div class="header-subtitle">H4 Spatial Certification | RSCT Geo-Cert | {timestamp}</div>
  </div>
  <div class="header-badge">kappa_spatial FAIL</div>
</div>

<!-- DASHBOARD GRID -->
<div class="dashboard">

  <!-- LEFT: KPI CARDS -->
  <div class="kpi-panel">

    <div class="kpi-card alert">
      <div class="kpi-label">Total Buildings Assessed</div>
      <div class="kpi-value blue">{fmt_num(kpis['total_buildings'])}</div>
      <div class="kpi-detail">Floodcaster model output</div>
    </div>

    <div class="kpi-card alert">
      <div class="kpi-label">Predicted Total Loss</div>
      <div class="kpi-value red">{fmt_money(kpis['total_pred_loss'])}</div>
      <div class="kpi-detail">Aggregated BldgLossUSD</div>
    </div>

    <div class="kpi-card ok">
      <div class="kpi-label">NFIP Historical Claims</div>
      <div class="kpi-value green">{fmt_money(kpis['total_nfip'])}</div>
      <div class="kpi-detail">1978-2023 paid losses</div>
    </div>

    <div class="kpi-card warn">
      <div class="kpi-label">Coverage Gap</div>
      <div class="kpi-value orange">{fmt_money(kpis['total_pred_loss'] - kpis['total_nfip'])}</div>
      <div class="kpi-detail">Predicted - Historical</div>
    </div>

    <div class="kpi-card alert">
      <div class="kpi-label">Kappa Spatial</div>
      <div class="kpi-value red">{kpis['kappa_spatial']:.3f}</div>
      <div class="kpi-detail">Threshold: 0.700 | FAIL</div>
    </div>

    <div class="kpi-card warn">
      <div class="kpi-label">Moran's I</div>
      <div class="kpi-value orange">{kpis['morans_i']:.3f}</div>
      <div class="kpi-detail">Negative = spatial dispersion</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-label">ZCTAs Analyzed</div>
      <div class="kpi-value">{kpis['n_zctas']}</div>
      <div class="kpi-detail">968xx prefix (Oahu)</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-label">Max Flood Depth</div>
      <div class="kpi-value">{kpis['max_depth_ft']:.1f} ft</div>
      <div class="kpi-detail">Pct inundated: {kpis['pct_inundated']:.1f}%</div>
    </div>

  </div>

  <!-- CENTER: MAP -->
  <div class="map-container">
    <div id="map"></div>

    <div class="map-overlay">
      <div class="legend-title">ZCTA Residual (Pred - NFIP)</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#2166ac;"></div> Underprediction</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#f4a582;"></div> Slight overprediction</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#b2182b;"></div> Major overprediction</div>
      <div class="legend-title" style="margin-top:10px;">Building Flood Depth</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ffffb2;"></div> 0 - 2 ft</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#fd8d3c;"></div> 2 - 8 ft</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#bd0026;"></div> 8+ ft</div>
    </div>
  </div>

  <!-- RIGHT: CHARTS -->
  <div class="charts-panel">

    <div class="verdict-banner">
      <div class="verdict-icon">X</div>
      <div class="verdict-text">
        <strong>SPATIAL CERTIFICATION: FAIL</strong><br>
        kappa_spatial = 0.352 &lt; 0.700 threshold.<br>
        Model predictions lack spatial coherence with historical claims.
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Predicted Loss vs NFIP Claims by ZCTA</div>
      <div class="chart-canvas-wrap">
        <canvas id="barChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Residual Distribution (Pred - NFIP)</div>
      <div class="chart-canvas-wrap">
        <canvas id="residualChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Top Mismatch ZCTAs</div>
      <table class="stats-table">
        <thead>
          <tr><th>ZCTA</th><th>Predicted</th><th>NFIP</th><th>Gap</th></tr>
        </thead>
        <tbody id="mismatchTable"></tbody>
      </table>
    </div>

  </div>

</div>

<!-- FOOTER -->
<div class="footer">
  <span>RSCT Geo-Cert | H4 Spatial Autocorrelation | Oahu, Hawaii</span>
  <span>Generated: {timestamp} UTC | Data: Floodcaster v0.3 + NFIP 1978-2023</span>
</div>

<script>
// ===== DATA =====
const zctaData = {zcta_geojson};
const buildingData = {buildings_geojson};
const barLabels = {bar_labels};
const barPred = {bar_pred};
const barNfip = {bar_nfip};

// ===== MAP =====
const map = L.map('map', {{
  center: [21.46, -157.97],
  zoom: 11,
  zoomControl: true,
  attributionControl: false,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 18,
  subdomains: 'abcd',
}}).addTo(map);

// Residual color scale (RdBu diverging)
function residualColor(val) {{
  const absVal = Math.abs(val);
  if (val < -5e6) return '#2166ac';
  if (val < -1e6) return '#4393c3';
  if (val < 0)    return '#92c5de';
  if (val < 1e6)  return '#f4a582';
  if (val < 5e6)  return '#d6604d';
  return '#b2182b';
}}

// ZCTA choropleth
function fmtMoney(v) {{
  if (Math.abs(v) >= 1e6) return '$' + (v/1e6).toFixed(1) + 'M';
  if (Math.abs(v) >= 1e3) return '$' + (v/1e3).toFixed(0) + 'K';
  return '$' + v.toFixed(0);
}}

L.geoJSON(zctaData, {{
  style: function(feature) {{
    return {{
      fillColor: residualColor(feature.properties.residual),
      fillOpacity: 0.55,
      color: '#58a6ff',
      weight: 1.5,
      opacity: 0.7,
    }};
  }},
  onEachFeature: function(feature, layer) {{
    const p = feature.properties;
    layer.bindPopup(
      '<div class="popup-zcta">ZCTA ' + p.zcta + '</div>' +
      '<div class="popup-row"><span class="popup-label">Predicted Loss</span><span class="popup-val">' + fmtMoney(p.pred_total_loss) + '</span></div>' +
      '<div class="popup-row"><span class="popup-label">NFIP Claims</span><span class="popup-val">' + fmtMoney(p.nfip_total) + '</span></div>' +
      '<div class="popup-row"><span class="popup-label">Residual</span><span class="popup-val" style="color:' + (p.residual > 0 ? '#e63946' : '#2ea043') + '">' + fmtMoney(p.residual) + '</span></div>' +
      '<div class="popup-row"><span class="popup-label">Buildings</span><span class="popup-val">' + p.n_buildings.toLocaleString() + '</span></div>'
    );
  }}
}}).addTo(map);

// Building flood depth color
function depthColor(d) {{
  if (d <= 0) return '#ffffb2';
  if (d <= 2) return '#fecc5c';
  if (d <= 5) return '#fd8d3c';
  if (d <= 8) return '#f03b20';
  return '#bd0026';
}}

L.geoJSON(buildingData, {{
  pointToLayer: function(feature, latlng) {{
    return L.circleMarker(latlng, {{
      radius: 2.5,
      fillColor: depthColor(feature.properties.depth_ft),
      fillOpacity: 0.7,
      color: 'none',
      weight: 0,
    }});
  }},
  onEachFeature: function(feature, layer) {{
    const p = feature.properties;
    layer.bindPopup(
      '<div class="popup-row"><span class="popup-label">Flood Depth</span><span class="popup-val">' + p.depth_ft + ' ft</span></div>' +
      '<div class="popup-row"><span class="popup-label">Building Loss</span><span class="popup-val">' + fmtMoney(p.loss_usd) + '</span></div>'
    );
  }}
}}).addTo(map);

// ===== CHARTS =====
const chartDefaults = {{
  color: '#e0e0e0',
  borderColor: '#21262d',
}};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

// Bar chart: Pred vs NFIP
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: barLabels,
    datasets: [
      {{
        label: 'Predicted Loss',
        data: barPred.map(v => v / 1e6),
        backgroundColor: '#e63946cc',
        borderColor: '#e63946',
        borderWidth: 1,
      }},
      {{
        label: 'NFIP Claims',
        data: barNfip.map(v => v / 1e6),
        backgroundColor: '#2ea043cc',
        borderColor: '#2ea043',
        borderWidth: 1,
      }},
    ],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ boxWidth: 12, padding: 8 }} }},
    }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 45 }} }},
      y: {{
        title: {{ display: true, text: 'USD (Millions)', font: {{ size: 11 }} }},
        ticks: {{ callback: v => '$' + v + 'M' }},
      }},
    }},
  }},
}});

// Residual distribution (horizontal bar)
const residuals = barPred.map((p, i) => (p - barNfip[i]) / 1e6);
const resColors = residuals.map(r => r > 0 ? '#e63946cc' : '#2ea043cc');

new Chart(document.getElementById('residualChart'), {{
  type: 'bar',
  data: {{
    labels: barLabels,
    datasets: [{{
      label: 'Residual ($M)',
      data: residuals,
      backgroundColor: resColors,
      borderColor: resColors.map(c => c.replace('cc', '')),
      borderWidth: 1,
    }}],
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Residual ($M)', font: {{ size: 11 }} }},
        ticks: {{ callback: v => '$' + v + 'M' }},
      }},
      y: {{ ticks: {{ font: {{ size: 9 }} }} }},
    }},
  }},
}});

// Mismatch table
const tableBody = document.getElementById('mismatchTable');
const indices = residuals.map((r, i) => [Math.abs(r), i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
indices.forEach(([absR, i]) => {{
  const tr = document.createElement('tr');
  const gap = barPred[i] - barNfip[i];
  tr.innerHTML = '<td>' + barLabels[i] + '</td>' +
    '<td>' + fmtMoney(barPred[i]) + '</td>' +
    '<td>' + fmtMoney(barNfip[i]) + '</td>' +
    '<td style="color:' + (gap > 0 ? '#e63946' : '#2ea043') + '">' + fmtMoney(gap) + '</td>';
  tableBody.appendChild(tr);
}});
</script>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Oahu Flood Risk Dashboard ===")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory() as tmp:
        # Load data
        buildings = load_buildings(tmp)
        residuals = load_residuals(tmp)
        zcta_gdf = load_zcta_boundaries(tmp)

        # Generate GeoJSON
        log.info("Converting to GeoJSON...")
        zcta_geojson = zcta_to_geojson(zcta_gdf, residuals)
        buildings_geojson = buildings_to_geojson(buildings, sample_n=5000)

        # Compute KPIs
        kpis = compute_kpis(buildings, residuals)
        log.info("KPIs: %s", {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in kpis.items()})

        # Build HTML
        log.info("Building dashboard HTML...")
        html = build_dashboard_html(
            zcta_geojson=zcta_geojson,
            buildings_geojson=buildings_geojson,
            kpis=kpis,
            residuals=residuals,
            timestamp=timestamp,
        )

        # Write locally
        out_dir = Path("/opt/ml/processing/output") if Path("/opt/ml").exists() else Path("outputs/oahu_dashboard")
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"oahu_flood_dashboard_{timestamp}.html"
        out_path = out_dir / fname
        out_path.write_text(html, encoding="utf-8")
        log.info("Dashboard written: %s (%d KB)", out_path, len(html) // 1024)

        # Upload to S3
        s3 = _s3()
        s3_key = f"geocert-experiments/s036/oahu_dashboard/{fname}"
        s3.upload_file(
            str(out_path),
            "swarm-yrsn-datasets",
            s3_key,
            ExtraArgs={"ContentType": "text/html"},
        )
        log.info("Uploaded to s3://swarm-yrsn-datasets/%s", s3_key)

        # Also upload a "latest" copy for easy access
        s3.upload_file(
            str(out_path),
            "swarm-yrsn-datasets",
            "geocert-experiments/s036/oahu_dashboard/latest.html",
            ExtraArgs={"ContentType": "text/html"},
        )
        log.info("Uploaded latest.html alias")

    log.info("=== Dashboard complete ===")


if __name__ == "__main__":
    main()
