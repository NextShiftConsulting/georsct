"""
validate_spatial_block_size.py -- Variogram-based spatial block size validation.

Validates that county-level spatial blocking matches the empirical
autocorrelation range of each target variable. If the autocorrelation
range exceeds county diameter, blocking is too fine (leakage risk).
If it's smaller, blocking may waste data.

Design inspired by R's blockCV package (clean-room, no GPL code reused).

Outputs:
    - Empirical variogram parameters (range, sill, nugget) per target
    - Comparison with actual block diameters
    - PASS/WARN/FAIL verdict per target

Usage:
    python validate_spatial_block_size.py --scenario houston --upload
    python validate_spatial_block_size.py --all-scenarios --upload
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
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, OUTPUT_KEYS, get_s3_client

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

# Targets to validate (the ones used in s035)
TARGETS = ["obs_nfip_event_claims", "obs_has_311", "obs_has_hwm"]


# ---------------------------------------------------------------------------
# Variogram fitting
# ---------------------------------------------------------------------------

def _spherical_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical variogram model."""
    result = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        sill,
    )
    result[h == 0] = 0.0
    return result


def compute_empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_bins: int = 15,
    max_dist_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical variogram from coordinates and values.

    Args:
        coords: (n, 2) array of coordinates (lon, lat or projected).
        values: (n,) array of target values.
        n_bins: Number of distance bins.
        max_dist_frac: Maximum distance as fraction of data extent.

    Returns:
        (bin_centers, semivariance) arrays.
    """
    # Compute pairwise distances (in coordinate units)
    dists = pdist(coords)
    max_dist = np.percentile(dists, max_dist_frac * 100)

    # Compute pairwise squared differences
    n = len(values)
    sq_diffs = pdist(values.reshape(-1, 1), metric="sqeuclidean")

    # Bin by distance
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    semivariance = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if mask.sum() > 0:
            semivariance[i] = np.mean(sq_diffs[mask]) / 2.0

    valid = np.isfinite(semivariance)
    return bin_centers[valid], semivariance[valid]


def fit_variogram(
    bin_centers: np.ndarray, semivariance: np.ndarray,
) -> dict:
    """Fit a spherical variogram model and return parameters.

    Returns:
        Dict with keys: nugget, sill, range_km, fit_r2
    """
    if len(bin_centers) < 3:
        return {"nugget": np.nan, "sill": np.nan, "range_km": np.nan, "fit_r2": np.nan}

    # Initial guesses
    nugget_0 = float(semivariance[0]) if semivariance[0] > 0 else 0.0
    sill_0 = float(np.max(semivariance))
    range_0 = float(bin_centers[len(bin_centers) // 2])

    try:
        popt, _ = curve_fit(
            _spherical_model,
            bin_centers, semivariance,
            p0=[nugget_0, sill_0, range_0],
            bounds=([0, 0, 0], [sill_0 * 2, sill_0 * 3, bin_centers[-1] * 2]),
            maxfev=5000,
        )
        nugget, sill, range_ = popt

        # R-squared of fit
        predicted = _spherical_model(bin_centers, *popt)
        ss_res = np.sum((semivariance - predicted) ** 2)
        ss_tot = np.sum((semivariance - np.mean(semivariance)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "nugget": float(nugget),
            "sill": float(sill),
            "range_km": float(range_),
            "fit_r2": float(r2),
        }
    except (RuntimeError, ValueError) as e:
        log.warning("Variogram fit failed: %s", e)
        return {"nugget": np.nan, "sill": np.nan, "range_km": np.nan, "fit_r2": np.nan}


# ---------------------------------------------------------------------------
# Block size estimation
# ---------------------------------------------------------------------------

def estimate_block_diameters(
    df: pd.DataFrame, fold_col: str = "fold_spatial_blocked",
) -> dict[str, float]:
    """Estimate the diameter of each spatial block (fold) in km.

    Uses ZCTA centroids and computes max pairwise distance within each fold.
    Requires lat/lon columns.
    """
    if fold_col not in df.columns:
        return {}

    diameters = {}
    for fold_id, group in df.groupby(fold_col):
        if len(group) < 2:
            diameters[str(fold_id)] = 0.0
            continue
        # Use lat/lon if available
        if "latitude" in group.columns and "longitude" in group.columns:
            coords = group[["longitude", "latitude"]].values
        elif "lon" in group.columns and "lat" in group.columns:
            coords = group[["lon", "lat"]].values
        else:
            diameters[str(fold_id)] = np.nan
            continue
        # Approximate km using Haversine-like scaling
        lat_mid = np.mean(coords[:, 1])
        km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mid))
        km_per_deg_lat = 110.57
        km_coords = coords.copy()
        km_coords[:, 0] *= km_per_deg_lon
        km_coords[:, 1] *= km_per_deg_lat
        dists = pdist(km_coords)
        diameters[str(fold_id)] = float(np.max(dists)) if len(dists) > 0 else 0.0

    return diameters


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_block_size(
    variogram_range_km: float,
    block_diameters: dict[str, float],
) -> dict:
    """Compare autocorrelation range to block diameters.

    Returns verdict dict with status and explanation.
    """
    if np.isnan(variogram_range_km):
        return {
            "status": "SKIP",
            "message": "variogram range could not be estimated",
        }

    valid_diameters = [d for d in block_diameters.values() if not np.isnan(d) and d > 0]
    if not valid_diameters:
        return {
            "status": "SKIP",
            "message": "no valid block diameters",
        }

    min_diameter = min(valid_diameters)
    median_diameter = float(np.median(valid_diameters))
    max_diameter = max(valid_diameters)

    ratio = variogram_range_km / median_diameter

    if ratio > 1.5:
        return {
            "status": "WARN",
            "message": (
                f"autocorrelation range ({variogram_range_km:.0f} km) > 1.5x "
                f"median block diameter ({median_diameter:.0f} km). "
                "Spatial leakage risk: consider larger blocks."
            ),
            "range_km": variogram_range_km,
            "median_block_km": median_diameter,
            "ratio": ratio,
        }
    elif ratio < 0.3:
        return {
            "status": "WARN",
            "message": (
                f"autocorrelation range ({variogram_range_km:.0f} km) < 0.3x "
                f"median block diameter ({median_diameter:.0f} km). "
                "Blocks may be unnecessarily large (wasting data)."
            ),
            "range_km": variogram_range_km,
            "median_block_km": median_diameter,
            "ratio": ratio,
        }
    else:
        return {
            "status": "PASS",
            "message": (
                f"autocorrelation range ({variogram_range_km:.0f} km) is "
                f"{ratio:.1f}x median block diameter ({median_diameter:.0f} km). "
                "Block size is appropriate."
            ),
            "range_km": variogram_range_km,
            "median_block_km": median_diameter,
            "ratio": ratio,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(s3, scenario: str, upload: bool = False) -> dict:
    """Run spatial block size validation for one scenario."""
    # Load assembled parquet
    key = OUTPUT_KEYS[scenario]
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return {"scenario": scenario, "status": "FAIL", "message": str(e)}

    # Need centroid coordinates -- check for lat/lon columns
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat", "centroid_lat")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "centroid_lon")), None)
    if lat_col is None or lon_col is None:
        return {
            "scenario": scenario,
            "status": "SKIP",
            "message": f"no lat/lon columns found (have: {list(df.columns)[:20]}...)",
        }

    # Convert to km coordinates
    lat_mid = df[lat_col].mean()
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mid))
    km_per_deg_lat = 110.57
    coords_km = np.column_stack([
        df[lon_col].values * km_per_deg_lon,
        df[lat_col].values * km_per_deg_lat,
    ])

    results = {"scenario": scenario, "timestamp": datetime.now(timezone.utc).isoformat()}
    target_results = {}

    for target in TARGETS:
        if target not in df.columns:
            target_results[target] = {"status": "SKIP", "message": "column not present"}
            continue

        values = df[target].values.astype(np.float64)
        valid = np.isfinite(values)
        if valid.sum() < 20:
            target_results[target] = {
                "status": "SKIP",
                "message": f"only {valid.sum()} non-null values",
            }
            continue

        # Compute variogram on valid subset
        bins, semivar = compute_empirical_variogram(
            coords_km[valid], values[valid],
        )
        vario_params = fit_variogram(bins, semivar)

        target_results[target] = {
            "variogram": vario_params,
            "n_valid": int(valid.sum()),
        }
        log.info(
            "%s / %s: range=%.1f km, sill=%.4f, R2=%.3f",
            scenario, target,
            vario_params["range_km"], vario_params["sill"], vario_params["fit_r2"],
        )

    # Load fold assignments if available
    fold_key = f"folds/{scenario}_folds.parquet"
    block_diameters = {}
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=fold_key)
        folds_df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        # Merge with coordinates
        if "zcta_id" in folds_df.columns and "zcta_id" in df.columns:
            merged = df[["zcta_id", lon_col, lat_col]].merge(folds_df, on="zcta_id")
            merged = merged.rename(columns={lon_col: "longitude", lat_col: "latitude"})
            block_diameters = estimate_block_diameters(merged)
            log.info("Block diameters: %s", {k: f"{v:.0f} km" for k, v in block_diameters.items()})
    except Exception:
        log.info("No fold assignments found -- skipping block diameter comparison")

    # Validate each target
    for target, tres in target_results.items():
        if "variogram" in tres:
            verdict = validate_block_size(tres["variogram"]["range_km"], block_diameters)
            tres["verdict"] = verdict

    results["targets"] = target_results
    results["block_diameters_km"] = block_diameters
    results["status"] = "COMPLETE"

    if upload:
        json_key = f"results/s035/spatial_block_validation_{scenario}.json"
        s3.put_object(
            Bucket=BUCKET, Key=json_key,
            Body=json.dumps(results, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info("Uploaded %s", json_key)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial block size validation")
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios:
        parser.error("specify --scenario or --all-scenarios")

    s3 = get_s3_client()
    scenarios = MODELABLE if args.all_scenarios else [args.scenario]

    for scenario in scenarios:
        log.info("=== Block size validation: %s ===", scenario)
        result = run_validation(s3, scenario, args.upload)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
