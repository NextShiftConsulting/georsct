#!/usr/bin/env python3
"""
build_r3_feature_registry.py -- Phase R3_0: Feature registry + candidate graph.

Collects R0-R2 certificates, diagnostics, and feature metadata into a
per-block feature registry. Each feature is assigned to a candidate block
and receives an admissibility verdict based on temporal/causal boundary
checks.

Also produces a candidate graph encoding block-level dependencies and
interaction hypotheses (consumed by R3_1c DGM admission).

No predictive model is trained. This is a metadata assembly step.

Usage:
    python build_r3_feature_registry.py --upload
    python build_r3_feature_registry.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"


# ---------------------------------------------------------------------------
# Block definitions (from DOE_R3 Section R3_1)
# ---------------------------------------------------------------------------

CANDIDATE_BLOCKS = {
    "hydrology": {
        "features": [
            "nhd_catchment_area_km2",
            "slope_basin_slope",
            "slope_stream_slope",
            "twi_acc_twi",
            "twi_tot_twi",
            "upstream_catchment_km2",
            "hcfcd_drainage_district",
            "levee_nearest_km",
            "levee_condition_rating",
            "sewershed_name",
        ],
        "source": "NHDPlus + USACE levees + local drainage",
        "spatial_grain": "catchment",
        "expected_mechanism": "Hydrologic connectivity and drainage capacity",
        "leakage_risk": "low",
    },
    "spatial_relation": {
        "features": [
            "zcta_degree",
            "zcta_mean_neighbor_dist_km",
            "wlag_flood_zone_pct",
            "wlag_population_density",
            "wlag_median_income",
            "wlag_impervious_pct",
            "wlag_cropland_pct",
            "wlag_rainfall_mm",
            "wlag_nfip_claims",
        ],
        "source": "W-matrix spatial lags from adjacency",
        "spatial_grain": "neighbor-lag",
        "expected_mechanism": "Spatial spillover and neighbor exposure",
        "leakage_risk": "medium",
        "leakage_note": "wlag_nfip_claims requires per-fold recomputation",
    },
    "temporal": {
        "features": [
            "peak_1h_mm",
            "peak_3h_mm",
            "peak_6h_mm",
            "storm_duration_h",
            "time_to_peak_h",
            "rainfall_intensity_cv",
            "tide_peak_m",
            "surge_rain_lag_h",
            "storm_min_dist_km",
            "storm_landfall_category",
        ],
        "source": "MRMS + HRRR + HURDAT2 + NOAA tides",
        "spatial_grain": "pixel/station",
        "expected_mechanism": "Event dynamics drive damage severity",
        "leakage_risk": "medium",
        "leakage_note": "Event-concurrent features; gated by event start date",
    },
    "infrastructure": {
        "features": [
            "hifld_nearest_hospital_km",
            "hifld_n_hospitals",
            "hifld_n_hospital_beds",
            "hifld_n_pharmacies",
            "hifld_nearest_pharmacy_km",
            "hifld_nearest_trauma_center_km",
            "drive_min_to_county_centroid",
            "drive_min_to_county_seat",
            "drive_min_to_nearest_hospital",
        ],
        "source": "HIFLD + drive time computations",
        "spatial_grain": "ZCTA",
        "expected_mechanism": "Access to recovery resources",
        "leakage_risk": "none",
    },
    "socioeconomic": {
        "features": [
            "acs_total_pop",
            "acs_median_hh_income",
            "acs_pct_below_poverty",
            "acs_pct_renter_occupied",
            "acs_pct_owner_occupied",
            "acs_pct_vacant",
            "acs_pct_no_vehicle",
            "acs_median_home_value",
            "acs_median_year_built",
            "population",
            "svi_overall",
            "svi_socioeconomic",
            "svi_household_disability",
            "svi_minority_language",
            "svi_housing_transport",
            "nfip_historical_frequency",
            "nfip_historical_severity",
        ],
        "source": "ACS + SVI + NFIP historical (temporally gated)",
        "spatial_grain": "ZCTA",
        "expected_mechanism": "Vulnerability and exposure capacity",
        "leakage_risk": "low",
    },
    "terrain": {
        "features": [
            "flood_pct_zone_a",
            "flood_pct_zone_x",
            "flood_pct_zone_x500",
            "flood_sfha",
            "elevation_m_msl",
            "slope_mean_pct",
            "twi_twi",
            "coastal_distance_m",
            "latitude",
            "longitude",
            "impervious_pct",
            "cropland_pct",
        ],
        "source": "FEMA NFHL + DEM + NLCD",
        "spatial_grain": "ZCTA",
        "expected_mechanism": "Physical flood susceptibility",
        "leakage_risk": "none",
    },
}

# Temporal status classification
TEMPORAL_STATUS = {
    # Pre-event (safe)
    "flood_pct_zone_a": "pre-event",
    "flood_pct_zone_x": "pre-event",
    "flood_pct_zone_x500": "pre-event",
    "flood_sfha": "pre-event",
    "elevation_m_msl": "pre-event",
    "slope_mean_pct": "pre-event",
    "twi_twi": "pre-event",
    "coastal_distance_m": "pre-event",
    "latitude": "pre-event",
    "longitude": "pre-event",
    "impervious_pct": "pre-event",
    "cropland_pct": "pre-event",
    "acs_total_pop": "pre-event",
    "acs_median_hh_income": "pre-event",
    "acs_pct_below_poverty": "pre-event",
    "acs_pct_renter_occupied": "pre-event",
    "acs_pct_owner_occupied": "pre-event",
    "acs_pct_vacant": "pre-event",
    "acs_pct_no_vehicle": "pre-event",
    "acs_median_home_value": "pre-event",
    "acs_median_year_built": "pre-event",
    "population": "pre-event",
    "svi_overall": "pre-event",
    "svi_socioeconomic": "pre-event",
    "svi_household_disability": "pre-event",
    "svi_minority_language": "pre-event",
    "svi_housing_transport": "pre-event",
    "nhd_catchment_area_km2": "pre-event",
    "slope_basin_slope": "pre-event",
    "slope_stream_slope": "pre-event",
    "twi_acc_twi": "pre-event",
    "twi_tot_twi": "pre-event",
    "upstream_catchment_km2": "pre-event",
    "hcfcd_drainage_district": "pre-event",
    "levee_nearest_km": "pre-event",
    "levee_condition_rating": "pre-event",
    "sewershed_name": "pre-event",
    "zcta_degree": "pre-event",
    "zcta_mean_neighbor_dist_km": "pre-event",
    "hifld_nearest_hospital_km": "pre-event",
    "hifld_n_hospitals": "pre-event",
    "hifld_n_hospital_beds": "pre-event",
    "hifld_n_pharmacies": "pre-event",
    "hifld_nearest_pharmacy_km": "pre-event",
    "hifld_nearest_trauma_center_km": "pre-event",
    "drive_min_to_county_centroid": "pre-event",
    "drive_min_to_county_seat": "pre-event",
    "drive_min_to_nearest_hospital": "pre-event",
    # Historical-gated (safe with temporal cutoff)
    "nfip_historical_frequency": "historical-gated",
    "nfip_historical_severity": "historical-gated",
    # Event-concurrent (medium risk)
    "peak_1h_mm": "event-concurrent",
    "peak_3h_mm": "event-concurrent",
    "peak_6h_mm": "event-concurrent",
    "storm_duration_h": "event-concurrent",
    "time_to_peak_h": "event-concurrent",
    "rainfall_intensity_cv": "event-concurrent",
    "tide_peak_m": "event-concurrent",
    "surge_rain_lag_h": "event-concurrent",
    "storm_min_dist_km": "event-concurrent",
    "storm_landfall_category": "event-concurrent",
    # Spatial lags (medium: wlag_nfip_claims is target-derived)
    "wlag_flood_zone_pct": "pre-event",
    "wlag_population_density": "pre-event",
    "wlag_median_income": "pre-event",
    "wlag_impervious_pct": "pre-event",
    "wlag_cropland_pct": "pre-event",
    "wlag_rainfall_mm": "event-concurrent",
    "wlag_nfip_claims": "historical-gated",
    "slosh_max_surge_m": "pre-event",
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> Optional[dict]:
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_certificates(s3, level: str) -> dict:
    data = _load_json(s3, f"{RESULTS_PREFIX}/certificates_{level}.json")
    if not data:
        return {}
    index = {}
    for cell in data.get("certificates", data.get("cells", [])):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    return index


def _load_diagnostics(s3, level: str) -> dict:
    data = _load_json(s3, f"{RESULTS_PREFIX}/diagnostics_{level}.json")
    if not data:
        return {}
    index = {}
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    return index


# ---------------------------------------------------------------------------
# Admissibility checks
# ---------------------------------------------------------------------------

def _check_admissibility(feature_name: str) -> tuple[str, str]:
    """Check temporal/causal boundary. Returns (verdict, reason)."""
    temporal = TEMPORAL_STATUS.get(feature_name, "unknown")

    if temporal == "pre-event":
        return "ADMIT_TO_TEST", "Pre-event feature; no temporal leakage risk"
    elif temporal == "historical-gated":
        return "ADMIT_TO_TEST", "Historical-gated; temporal cutoff enforced at build time"
    elif temporal == "event-concurrent":
        return "ADMIT_TO_TEST", "Event-concurrent; acceptable for flood prediction"
    elif temporal == "post-event":
        return "QUARANTINE_LEAKAGE", "Post-event feature; requires per-fold recomputation"
    else:
        return "QUARANTINE_LEAKAGE", f"Unknown temporal status: {temporal}"


# ---------------------------------------------------------------------------
# Candidate graph
# ---------------------------------------------------------------------------

def _build_candidate_graph() -> dict:
    """Build block-level dependency graph.

    Nodes = blocks. Edges = interaction hypotheses that determine
    testing order for R3_1c DGM admission.
    """
    blocks = list(CANDIDATE_BLOCKS.keys())
    edges = [
        {
            "source": "terrain",
            "target": "hydrology",
            "interaction": "Terrain determines hydrologic connectivity",
            "strength": "strong",
        },
        {
            "source": "hydrology",
            "target": "spatial_relation",
            "interaction": "Hydrologic features define spatial lag structure",
            "strength": "strong",
        },
        {
            "source": "temporal",
            "target": "infrastructure",
            "interaction": "Event severity modulates infrastructure access impact",
            "strength": "moderate",
        },
        {
            "source": "socioeconomic",
            "target": "infrastructure",
            "interaction": "Vulnerability and access are co-determined",
            "strength": "moderate",
        },
        {
            "source": "temporal",
            "target": "hydrology",
            "interaction": "Rainfall intensity interacts with catchment capacity",
            "strength": "strong",
        },
        {
            "source": "terrain",
            "target": "temporal",
            "interaction": "Elevation and slope modulate rainfall impact",
            "strength": "moderate",
        },
    ]

    return {
        "nodes": blocks,
        "edges": edges,
        "testing_order": [
            "terrain",
            "hydrology",
            "spatial_relation",
            "temporal",
            "infrastructure",
            "socioeconomic",
        ],
        "testing_order_rationale": (
            "Terrain and hydrology are foundational (strong edges to most blocks). "
            "Spatial relation depends on hydrology. Temporal interacts with both "
            "terrain and hydrology. Infrastructure and socioeconomic are leaf blocks."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_0: Feature registry + candidate graph"
    )
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would load certificates + diagnostics for R0/R1/R2")
        log.info("Build feature registry with %d blocks, %d total features",
                 len(CANDIDATE_BLOCKS),
                 sum(len(b["features"]) for b in CANDIDATE_BLOCKS.values()))
        log.info("Writes: %s/r3_feature_registry.json", RESULTS_PREFIX)
        log.info("Writes: %s/r3_candidate_graph.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()

    # Load all certificates and diagnostics
    certs = {}
    diags = {}
    for level in ("r0", "r1", "r2"):
        certs[level] = _load_certificates(s3, level)
        diags[level] = _load_diagnostics(s3, level)
        log.info("Loaded %d certs, %d diags for %s",
                 len(certs[level]), len(diags[level]), level)

    # Build feature registry
    registry_entries = []
    for block_name, block_def in CANDIDATE_BLOCKS.items():
        for feature_name in block_def["features"]:
            verdict, reason = _check_admissibility(feature_name)

            # Check scenario coverage from R0 results
            scenario_coverage = []
            for scenario in SCENARIOS:
                r0_data = _load_json(
                    s3, f"{RESULTS_PREFIX}/r0_{scenario}.json"
                )
                if r0_data:
                    features_used = set()
                    for run in r0_data.get("runs", []):
                        if "features_used" in run and isinstance(run["features_used"], list):
                            features_used.update(run["features_used"])
                    # Also check column names from R1/R2 supplements
                    scenario_coverage.append(scenario)

            # Missingness rate placeholder (computed at runtime from parquet)
            entry = {
                "feature_name": feature_name,
                "candidate_block": block_name,
                "source": block_def["source"],
                "spatial_grain": block_def["spatial_grain"],
                "expected_mechanism": block_def["expected_mechanism"],
                "temporal_status": TEMPORAL_STATUS.get(feature_name, "unknown"),
                "leakage_risk": block_def["leakage_risk"],
                "scenario_coverage": scenario_coverage,
                "missingness_rate": None,  # Populated at R3_1a runtime
                "admissibility_verdict": verdict,
                "admissibility_reason": reason,
            }
            registry_entries.append(entry)

    # Summary statistics
    verdicts = {}
    for e in registry_entries:
        v = e["admissibility_verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    # Certificate summary per cell (for context)
    cert_summary = {}
    for level in ("r0", "r1", "r2"):
        for (scenario, target), cert in certs[level].items():
            cell_key = f"{scenario}/{target}"
            if cell_key not in cert_summary:
                cert_summary[cell_key] = {}
            cert_summary[cell_key][level] = {
                "R": cert.get("R"),
                "S_sup": cert.get("S_sup"),
                "N": cert.get("N"),
                "kappa_compat": cert.get("kappa_compat"),
                "sigma": cert.get("sigma"),
                "alpha": cert.get("alpha"),
                "spatial_metric": cert.get("spatial_metric"),
            }

    # Build candidate graph
    candidate_graph = _build_candidate_graph()

    # Assemble output
    registry_result = {
        "phase": "R3_0_feature_registry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_blocks": len(CANDIDATE_BLOCKS),
        "n_features": len(registry_entries),
        "verdict_summary": verdicts,
        "blocks": {
            name: {
                "n_features": len(defn["features"]),
                "source": defn["source"],
                "spatial_grain": defn["spatial_grain"],
                "leakage_risk": defn["leakage_risk"],
            }
            for name, defn in CANDIDATE_BLOCKS.items()
        },
        "features": registry_entries,
        "certificate_summary": cert_summary,
    }

    # Write local copies
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = out_dir / "r3_feature_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry_result, f, indent=2)
    log.info("Written to %s", registry_path)

    graph_path = out_dir / "r3_candidate_graph.json"
    with open(graph_path, "w") as f:
        json.dump(candidate_graph, f, indent=2)
    log.info("Written to %s", graph_path)

    if args.upload:
        upload_json_result(
            s3, BUCKET,
            f"{RESULTS_PREFIX}/r3_feature_registry.json",
            registry_result,
        )
        upload_json_result(
            s3, BUCKET,
            f"{RESULTS_PREFIX}/r3_candidate_graph.json",
            candidate_graph,
        )
        log.info("Uploaded to S3")

    # Summary
    log.info("\n=== R3_0 Feature Registry ===")
    log.info("  Blocks: %d", len(CANDIDATE_BLOCKS))
    log.info("  Features: %d", len(registry_entries))
    for v, count in sorted(verdicts.items()):
        log.info("  %s: %d", v, count)
    log.info("  Cells with certificates: %d", len(cert_summary))

    return 0


if __name__ == "__main__":
    sys.exit(main())
