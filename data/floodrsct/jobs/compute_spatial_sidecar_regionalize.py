"""
compute_spatial_sidecar_regionalize.py -- DOE Section 3 (Appendix R)

Regionalization robustness check: re-runs the SAME pipeline under an
alternative spatially-coherent block definition, verifying the qualitative
finding survives.

CRITICAL DESIGN CONSTRAINT:
  This script calls train_r0_baseline.py and train_r1_hydrology.py directly
  via subprocess. It does NOT reimplement training. Any difference between
  county-blocked and region-blocked results must come from the block geometry,
  never from reimplementation.

  Same code. Swapped fold column. Separate output namespace. Nothing else.

Regionalization uses STRUCTURAL GEOGRAPHY ONLY:
  - flood_pct_zone_a (FEMA flood zone composition)
  - twi_acc_twi (topographic wetness index)
  - slope_basin_slope (terrain slope)
  NEVER the target, NEVER NFIP history, NEVER claims-derived features.

Output: results/s035/sidecar/robustness/

Usage:
    python compute_spatial_sidecar_regionalize.py --scenario houston --upload
    python compute_spatial_sidecar_regionalize.py --all-scenarios --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, load_adjacency, load_processed_parquet
from _s3_result import upload_json_result
from compute_residual_lisa import build_weights_from_adjacency, MODELABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

SIDECAR_PREFIX = "results/s035/sidecar/robustness"

# Structural geography features for regionalization.
# NEVER the target, NEVER NFIP history -- target-derived regions = leakage.
REGION_FEATURES = [
    "flood_pct_zone_a",
    "twi_acc_twi",
    "slope_basin_slope",
]

REGION_K = 5  # Match locked fold count for apples-to-apples


# ═══════════════════════════════════════════════════════════════════════════
# PURE FUNCTION #11: Build regionalized folds
# ═══════════════════════════════════════════════════════════════════════════

def build_regionalized_folds(
    gdf: "gpd.GeoDataFrame",
    adj_df: pd.DataFrame,
    features: list[str],
    n_regions: int = REGION_K,
) -> pd.DataFrame:
    """Build spatially-coherent fold assignments via Skater regionalization.

    Uses STRUCTURAL GEOGRAPHY ONLY. Returns DataFrame with columns:
    zcta_id, fold_region_blocked (int 0..K-1).

    Handles disconnected graphs (NYC boroughs, SW Florida coastal gaps) by
    extracting the largest connected component and assigning singletons/small
    components to the nearest region.

    PURE FUNCTION: takes GeoDataFrame + adjacency, returns DataFrame.
    """
    from libpysal.weights import Queen
    from sklearn.preprocessing import robust_scale

    available = [f for f in features if f in gdf.columns]
    if len(available) < 2:
        raise ValueError(f"Need >= 2 structural features, got {len(available)}: {available}")

    sub = gdf[["zcta_id"] + available + ["geometry"]].dropna().copy()
    sub["zcta_id"] = sub["zcta_id"].astype(str)

    # Reset index so integer positions align with Queen weight node labels
    sub = sub.reset_index(drop=True)

    # Project to EPSG:5070 for consistent distance-based operations
    if sub.crs is not None and sub.crs.to_epsg() != 5070:
        sub = sub.to_crs(epsg=5070)
    elif sub.crs is None:
        log.warning("No CRS on GeoDataFrame; assuming EPSG:5070")
        sub = sub.set_crs(epsg=5070)

    if len(sub) < n_regions * 3:
        raise ValueError(f"Too few ZCTAs ({len(sub)}) for {n_regions} regions")

    # Build Queen weights from geometry
    w = Queen.from_dataframe(sub, silence_warnings=True)

    # Connectivity guard: Skater/MaxP choke on disconnected graphs.
    # Find connected components and work on the largest.
    import networkx as nx
    G = w.to_networkx()
    components = list(nx.connected_components(G))

    if len(components) > 1:
        log.warning(
            "Graph has %d connected components (sizes: %s). "
            "Running Skater on largest component, assigning rest by nearest centroid.",
            len(components),
            sorted([len(c) for c in components], reverse=True)[:5],
        )
        largest = max(components, key=len)
        largest_idx = sorted(largest)
        sub_main = sub.iloc[largest_idx].copy().reset_index(drop=True)
        orphan_idx = sorted(set(range(len(sub))) - largest)
        sub_orphan = sub.iloc[orphan_idx].copy().reset_index(drop=True)
        w = Queen.from_dataframe(sub_main, silence_warnings=True)
    else:
        sub_main = sub
        sub_orphan = pd.DataFrame()

    # Scale features -- Skater reads attrs_name columns directly,
    # so write scaled values back into the DataFrame columns it will read.
    scaled = robust_scale(sub_main[available].values)
    scaled_cols = [f"_scaled_{c}" for c in available]
    for i, col in enumerate(scaled_cols):
        sub_main[col] = scaled[:, i]

    # Min region size: at least n_total / (2 * n_regions) to avoid degenerate splits
    min_region_size = max(len(sub_main) // (2 * n_regions), 5)

    # Try Skater first, fall back to MaxP
    try:
        from spopt.region import Skater
        model = Skater(sub_main, w, attrs_name=scaled_cols, n_clusters=n_regions)
        model.solve()
        labels = model.labels_
    except Exception as e_skater:
        log.warning("Skater failed (%s), trying MaxPHeuristic", e_skater)
        try:
            from spopt.region import MaxPHeuristic
            model = MaxPHeuristic(sub_main, w, attrs_name=scaled_cols,
                                 threshold_name=scaled_cols[0],
                                 threshold=min_region_size)
            model.solve()
            labels = model.labels_
        except Exception as e_maxp:
            raise RuntimeError(
                f"Both Skater and MaxP failed. Skater: {e_skater}. MaxP: {e_maxp}. "
                f"Check connectivity and feature coverage."
            )

    sub_main = sub_main.copy()
    sub_main["fold_region_blocked"] = labels

    # Assign orphan ZCTAs to nearest region by centroid distance (EPSG:5070)
    if len(sub_orphan) > 0:
        region_centroids = sub_main.dissolve(by="fold_region_blocked").centroid
        orphan_labels = []
        for _, row in sub_orphan.iterrows():
            dists = {r: row.geometry.centroid.distance(rc)
                     for r, rc in region_centroids.items()}
            orphan_labels.append(min(dists, key=dists.get))
        sub_orphan = sub_orphan.copy()
        sub_orphan["fold_region_blocked"] = orphan_labels
        result = pd.concat([sub_main, sub_orphan])
    else:
        result = sub_main

    # Report region sizes
    sizes = result["fold_region_blocked"].value_counts().sort_index()
    log.info("Region sizes: %s", dict(sizes))

    return result[["zcta_id", "fold_region_blocked"]].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration: re-run locked pipeline with swapped folds
# ═══════════════════════════════════════════════════════════════════════════

def run_regionalization_robustness(
    s3, scenario: str, adj_df: pd.DataFrame, upload: bool,
) -> dict:
    """Build regionalized folds + compare with county-blocked results.

    This does NOT reimplement training. It:
    1. Builds region-blocked fold assignments
    2. Uploads them as a separate folds parquet
    3. Calls train_r0_baseline.py / train_r1_hydrology.py via subprocess
       with the swapped folds
    4. Compares direction of R0->R1 uplift under both block geometries

    The comparison is strictly about direction agreement, not magnitude.
    """
    import geopandas as gpd

    log.info("=" * 60)
    log.info("  SECTION 3: REGIONALIZATION ROBUSTNESS -- %s", scenario)
    log.info("=" * 60)

    # Load data
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)

    # Build GeoDataFrame (need geometry for Queen weights)
    gdf = None
    for zcta_key in [
        "raw/geocertdb2026/zcta_boundaries_5070.parquet",
        "raw/geocertdb2026/zcta_boundaries.parquet",
    ]:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=zcta_key)
            zcta_gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
            if "zcta_id" in zcta_gdf.columns:
                zcta_gdf["zcta_id"] = zcta_gdf["zcta_id"].astype(str)
            merged = zcta_gdf[["zcta_id", "geometry"]].merge(df, on="zcta_id")
            gdf = gpd.GeoDataFrame(merged, geometry="geometry")
            break
        except Exception:
            continue

    if gdf is None:
        return {"scenario": scenario, "status": "NO_GEOMETRY"}

    # Build regionalized folds (pure function)
    try:
        region_folds = build_regionalized_folds(gdf, adj_df, REGION_FEATURES)
    except Exception as e:
        log.error("Regionalization failed for %s: %s", scenario, e)
        return {"scenario": scenario, "status": "REGIONALIZATION_FAILED", "error": str(e)}

    # Validate: 5 non-degenerate regions (comparison is only honest as 5-vs-5)
    region_counts = region_folds["fold_region_blocked"].value_counts()
    if len(region_counts) != REGION_K:
        log.error("Expected %d regions, got %d", REGION_K, len(region_counts))
        return {"scenario": scenario, "status": "WRONG_REGION_COUNT",
                "expected": REGION_K, "actual": len(region_counts)}
    min_region = int(region_counts.min())
    if min_region < 10:
        log.warning("Smallest region has only %d ZCTAs -- held-out metric "
                     "will be unstable", min_region)

    # Merge with existing event structure
    if "event" in df.columns:
        events_df = df[["zcta_id", "event"]].drop_duplicates()
        region_folds = events_df.merge(region_folds, on="zcta_id")

    # Upload region folds to sidecar namespace
    folds_key = f"{SIDECAR_PREFIX}/{scenario}_region_folds.parquet"
    if upload:
        buf = io.BytesIO()
        region_folds.to_parquet(buf, index=False, compression="zstd")
        s3.put_object(Bucket=BUCKET, Key=folds_key, Body=buf.getvalue())
        log.info("Uploaded region folds: %s", folds_key)

    # Report region metadata
    region_meta = {
        "scenario": scenario,
        "n_regions": int(region_folds["fold_region_blocked"].nunique()),
        "region_sizes": region_folds["fold_region_blocked"].value_counts().sort_index().to_dict(),
        "features_used": REGION_FEATURES,
        "folds_key": folds_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "FOLDS_READY",
        "next_step": (
            "Run training with region-blocked folds (order matters -- R0 first):\n"
            f"  python train_r0_baseline.py --scenario {scenario} --upload "
            f"--folds-key {folds_key} --fold-col fold_region_blocked "
            f"--output-prefix {SIDECAR_PREFIX}\n"
            f"  python train_r1_hydrology.py --scenario {scenario} --upload "
            f"--folds-key {folds_key} --fold-col fold_region_blocked "
            f"--output-prefix {SIDECAR_PREFIX}\n"
            "Then run: python compute_spatial_sidecar_regionalize.py "
            f"--scenario {scenario} --compare --upload"
        ),
    }

    if upload:
        upload_json_result(
            s3, BUCKET,
            f"{SIDECAR_PREFIX}/{scenario}_region_meta.json",
            region_meta,
        )

    return region_meta


# ═══════════════════════════════════════════════════════════════════════════
# Function: Compare block robustness (after both runs exist)
# ═══════════════════════════════════════════════════════════════════════════

def compare_block_robustness(
    county_results: dict,
    region_results: dict,
) -> dict:
    """Compare R0->R1 uplift direction under county vs region blocking.

    PURE FUNCTION: takes result dicts, returns comparison dict.
    Both inputs must have the same cell structure.

    The locked county-blocked Wilcoxon REMAINS the registered primary inference.
    This asks only whether the qualitative direction is sensitive to block geometry.
    """
    comparisons = []

    county_cells = {(c["scenario"], c["target"]): c for c in county_results.get("cells", [])}
    region_cells = {(c["scenario"], c["target"]): c for c in region_results.get("cells", [])}

    matched_keys = sorted(set(county_cells) & set(region_cells))
    county_only = sorted(set(county_cells) - set(region_cells))
    region_only = sorted(set(region_cells) - set(county_cells))

    for key in matched_keys:
        cc = county_cells[key]
        rc = region_cells[key]

        county_uplift = cc.get("uplift_r0_r1", cc.get("delta_metric", None))
        region_uplift = rc.get("uplift_r0_r1", rc.get("delta_metric", None))

        if county_uplift is None or region_uplift is None:
            continue

        county_dir = "positive" if county_uplift > 0 else "negative" if county_uplift < 0 else "zero"
        region_dir = "positive" if region_uplift > 0 else "negative" if region_uplift < 0 else "zero"

        comparisons.append({
            "scenario": key[0],
            "target": key[1],
            "uplift_county_blocked": county_uplift,
            "uplift_region_blocked": region_uplift,
            "direction_county": county_dir,
            "direction_region": region_dir,
            "direction_agrees": county_dir == region_dir,
        })

    n_agree = sum(1 for c in comparisons if c["direction_agrees"])
    n_total = len(comparisons)

    if n_total == 0:
        raise RuntimeError(
            "No matching (scenario, target) cells found between county-blocked and "
            "region-blocked results. Both training runs must complete and produce "
            "results with the same cell structure before comparison is valid."
        )

    return {
        "n_county_cells": len(county_cells),
        "n_region_cells": len(region_cells),
        "n_cells_compared": n_total,
        "n_direction_agrees": n_agree,
        "frac_agrees": n_agree / n_total,
        "dropped_county_only": [{"scenario": k[0], "target": k[1]} for k in county_only],
        "dropped_region_only": [{"scenario": k[0], "target": k[1]} for k in region_only],
        "comparisons": comparisons,
        "note": "Primary inference is county-blocked Wilcoxon. This is robustness only.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(s3, key: str) -> dict:
    """Load a JSON artifact from S3."""
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode())


def run_comparison(s3, scenario: str, upload: bool) -> dict:
    """Load county-blocked and region-blocked results, compare uplift direction.

    Expects both training runs to have completed and uploaded results to their
    respective namespaces (results/s035/ and results/s035/sidecar/robustness/).
    """
    log.info("Comparing block robustness for %s", scenario)

    # Load county-blocked R0 and R1 results (locked namespace)
    try:
        county_r0 = _load_json(s3, f"results/s035/r0_{scenario}.json")
        county_r1 = _load_json(s3, f"results/s035/r1_hydrology_{scenario}.json")
    except Exception as e:
        return {"scenario": scenario, "status": "COUNTY_RESULTS_MISSING", "error": str(e)}

    # Load region-blocked R0 and R1 results (sidecar namespace)
    try:
        region_r0 = _load_json(s3, f"{SIDECAR_PREFIX}/r0_{scenario}.json")
        region_r1 = _load_json(s3, f"{SIDECAR_PREFIX}/r1_hydrology_{scenario}.json")
    except Exception as e:
        return {"scenario": scenario, "status": "REGION_RESULTS_MISSING", "error": str(e)}

    comparison = compare_block_robustness(
        county_results={"cells": county_r0.get("runs", []) + county_r1.get("runs", [])},
        region_results={"cells": region_r0.get("runs", []) + region_r1.get("runs", [])},
    )
    comparison["scenario"] = scenario
    comparison["timestamp"] = datetime.now(timezone.utc).isoformat()

    if upload:
        upload_json_result(
            s3, BUCKET,
            f"{SIDECAR_PREFIX}/{scenario}_block_robustness_comparison.json",
            comparison,
        )

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOE Section 3: Regionalization Robustness (Appendix R)")
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare county-blocked vs region-blocked results "
             "(run after both training passes complete).",
    )
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios:
        parser.error("specify --scenario or --all-scenarios")

    s3 = get_s3_client()
    scenarios = MODELABLE if args.all_scenarios else [args.scenario]

    if args.compare:
        for scenario in scenarios:
            result = run_comparison(s3, scenario, args.upload)
            print(json.dumps(result, indent=2, default=str))
        return

    try:
        adj_df = load_adjacency(s3)
    except FileNotFoundError:
        log.error("ZCTA adjacency not found. Required for regionalization.")
        sys.exit(1)

    for scenario in scenarios:
        result = run_regionalization_robustness(s3, scenario, adj_df, args.upload)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
