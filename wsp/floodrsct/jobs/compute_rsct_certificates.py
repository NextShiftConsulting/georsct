#!/usr/bin/env python3
"""
compute_rsct_certificates.py -- RSCT certificate sidecar + demos.

Thin CLI over rsct.experiment_cert.certify_experiment_cell() and
yrsn_controlplane.SequentialGatekeeper. No hand-rolled RSN, oobleck,
or gate logic -- delegates to the canonical packages.

Three modes:
  --collapse    Score-collapse demo: R/S/N vs geometry kappa gradient.
  --certificate Certificate sidecar per (scenario, target) cell.
  --sweep       Inflation-vs-sigma sweep: vary sigma multiplier and
                show how oobleck threshold changes decisions.

Data pipeline:
  1. Load pre-computed results from S3 (r0_{scenario}.json from Phase 1)
  2. Load pre-computed geometry_kappa.json from Phase 0.5
  3. Call certify_experiment_cell() for RSN + alpha/omega/tau/sigma
  4. Call SequentialGatekeeper.evaluate() for enforcement decision
  5. Emit JSON sidecar per mode

If pre-computed results are not available for a cell, falls back to
inline Ridge/LogisticRegression CV to produce fold metrics.

Inputs:  r0_*.json, geometry_kappa.json, assembled parquets (all on S3)
Outputs: rsct_{collapse,certificates,sweep}.json on S3

Usage:
    python compute_rsct_certificates.py --collapse --upload
    python compute_rsct_certificates.py --certificate --upload
    python compute_rsct_certificates.py --sweep --upload
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
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
from _s3_result import upload_json_result

# rsct service layer -- canonical RSN + certificate
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsct.experiment_cert import certify_experiment_cell

# yrsn controlplane -- canonical gate evaluation
try:
    from yrsn_controlplane import (
        PRESET_GEOSPATIAL_CONUS27 as _CFG,
        CPGatekeeperInput,
        SequentialGatekeeper,
        EnforcementDecision,
        tau_to_gear,
    )
    _GATEKEEPER = SequentialGatekeeper(_CFG)
    _HAS_CONTROLPLANE = True
except ImportError:
    _HAS_CONTROLPLANE = False
    _GATEKEEPER = None
    _CFG = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SEED = 42

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

TARGETS = [
    ("obs_nfip_event_claims", "regression", "log1p"),
    ("obs_has_311",           "classification", None),
    ("obs_has_hwm",           "classification", None),
]

# Valid (scenario, target) cells per EXPERIMENT_CONTRACT.yaml v1.2.
# Cells outside this map are degenerate (single-class or missing data).
VALID_CELLS = {
    "obs_nfip_event_claims": MODELABLE,  # all 5
    "obs_has_311":           ["houston", "nyc", "new_orleans"],
    "obs_has_hwm":           ["nyc", "southwest_florida"],
}

PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "roc_auc",
}

R0_FEATURES = [
    "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
    "elevation_m_msl", "slope_mean_pct", "twi_twi",
    "coastal_distance_m", "latitude", "longitude",
    "acs_total_pop", "acs_median_hh_income", "acs_pct_below_poverty",
    "acs_pct_renter_occupied", "acs_pct_owner_occupied", "acs_pct_vacant",
    "acs_pct_no_vehicle", "acs_median_home_value", "acs_median_year_built",
    "svi_overall", "svi_socioeconomic", "svi_household_disability",
    "svi_minority_language", "svi_housing_transport",
    "nfip_historical_frequency", "nfip_historical_severity",
    "hifld_nearest_hospital_km", "hifld_n_hospitals", "population",
    "impervious_pct", "cropland_pct",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def _load_adjacency(s3) -> pd.DataFrame:
    """Load ZCTA adjacency edge list from S3."""
    for key in [
        "raw/geocertdb2026/zcta_adjacency.parquet",
        "raw/geocert/zcta_adjacency.parquet",
    ]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            return pd.read_parquet(buf)
        except Exception:
            continue
    log.warning("zcta_adjacency.parquet not found; connectivity will be 0")
    return pd.DataFrame()


def _load_geometry_kappa(s3) -> dict:
    """Load Phase 0.5 geometry kappa, indexed by (scenario, target)."""
    data = _load_json(s3, f"{RESULTS_PREFIX}/geometry_kappa.json")
    if not data:
        log.warning(
            "geometry_kappa.json not found -- run compute_geometry_kappa.py "
            "first. kappa_geom will default to None."
        )
        return {}
    index = {}
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    log.info("Loaded geometry kappa for %d cells", len(index))
    return index


def _load_r0_results(s3, scenario: str) -> dict | None:
    """Load Phase 1 R0 results for a scenario."""
    return _load_json(s3, f"{RESULTS_PREFIX}/r0_{scenario}.json")


def _extract_fold_metrics(
    results: dict, target: str, split: str, solver: str,
) -> list[float]:
    """Extract per-fold primary metric values from a results JSON."""
    runs = results.get("runs", [])
    task_type = None
    for r in runs:
        if r["target"] == target:
            task_type = r.get("task")
            break
    if not task_type:
        return []

    metric_name = PRIMARY_METRIC.get(task_type)
    if not metric_name:
        return []

    vals = []
    for r in runs:
        if (r["target"] == target and r["split"] == split
                and r["solver"] == solver):
            v = r["metrics"].get(metric_name)
            if v is not None:
                vals.append(float(v))
    return vals


# ---------------------------------------------------------------------------
# Fallback: inline CV when pre-computed results are not available
# ---------------------------------------------------------------------------

def _fallback_cv(
    s3,
    scenario: str,
    target: str,
    task_type: str,
    transform: str | None,
) -> tuple[list[float], list[float]]:
    """Run inline Ridge/LogisticRegression CV as fallback.

    Returns (spatial_folds, random_folds). Since we don't have spatial-blocked
    folds here, spatial_folds = random_folds = standard 5-fold CV scores.
    S_sup will be 0 (no random-spatial gap computable).
    """
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    df = load_processed_parquet(s3, scenario)
    features = [f for f in R0_FEATURES if f in df.columns]

    if target not in df.columns:
        return [], []

    sub = df[features + [target]].dropna()
    if len(sub) < 20:
        return [], []

    X = sub[features].values
    y = sub[target].values.copy()

    if transform == "log1p":
        y = np.log1p(np.clip(y, 0, None))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    try:
        if task_type == "classification":
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            if n_pos < 5 or n_neg < 5:
                return [], []
            model = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                max_iter=1000, solver="lbfgs",
            )
            scoring = "roc_auc"
        else:
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            scoring = "r2"

        fold_scores = cross_val_score(model, X, y, cv=5, scoring=scoring,
                                       n_jobs=-1)
        return fold_scores.tolist(), fold_scores.tolist()
    except Exception as e:
        log.warning("Fallback CV failed for %s/%s: %s", scenario, target, e)
        return [], []


# ---------------------------------------------------------------------------
# Certificate builder using rsct.experiment_cert
# ---------------------------------------------------------------------------

def build_certificate(
    s3,
    scenario: str,
    target: str,
    task_type: str,
    transform: str | None,
    r0_results: dict | None,
    geometry_index: dict,
    sigma_inflation: float = 1.0,
) -> dict:
    """Build certificate for one (scenario, target) cell.

    Uses certify_experiment_cell() from rsct.experiment_cert and
    SequentialGatekeeper from yrsn_controlplane.
    """
    # Extract fold metrics from pre-computed results or fallback to CV
    if r0_results:
        spatial_folds = _extract_fold_metrics(
            r0_results, target, "spatial_blocked", "histgbdt",
        )
        random_folds = _extract_fold_metrics(
            r0_results, target, "random", "histgbdt",
        )
    else:
        spatial_folds, random_folds = [], []

    # Fallback to inline CV if no pre-computed results
    used_fallback = False
    if not spatial_folds:
        log.info("No pre-computed results for %s/%s, using fallback CV",
                 scenario, target)
        spatial_folds, random_folds = _fallback_cv(
            s3, scenario, target, task_type, transform,
        )
        used_fallback = True

    spatial_metric = float(np.mean(spatial_folds)) if spatial_folds else None
    random_metric = float(np.mean(random_folds)) if random_folds else None

    # Geometry kappa from Phase 0.5
    geom_cell = geometry_index.get((scenario, target), {})
    kappa_geom = geom_cell.get("kappa_geom")

    # Delegate to rsct service layer for RSN + alpha/omega/tau/sigma
    cert = certify_experiment_cell(
        spatial_metric=spatial_metric,
        random_metric=random_metric,
        task_type=task_type,
        fold_metrics=spatial_folds,
        kappa_geom=kappa_geom,
    )

    # Apply sigma inflation for sweep mode
    sigma = cert.sigma * sigma_inflation

    # Gate evaluation via controlplane (if available)
    gate_decision = None
    gear = None
    if _HAS_CONTROLPLANE and spatial_metric is not None:
        try:
            # Vendored wheel 0.1.0 uses kappa_gate; source renamed to
            # kappa_compat. Try both to survive wheel rebuild.
            kappa_val = cert.kappa_compat
            try:
                gate_input = CPGatekeeperInput(
                    alpha=cert.alpha,
                    kappa_gate=kappa_val,
                    sigma=sigma,
                    source_mode="proxy",
                    evidence={
                        "R": cert.R,
                        "S_sup": cert.S_sup,
                        "N": cert.N,
                        "omega": cert.omega,
                        "noise_admissibility": cert.N,
                        "proxy_domain": "geospatial_tabular",
                    },
                )
            except TypeError:
                gate_input = CPGatekeeperInput(
                    alpha=cert.alpha,
                    kappa_compat=kappa_val,
                    sigma=sigma,
                    source_mode="proxy",
                    evidence={
                        "R": cert.R,
                        "S_sup": cert.S_sup,
                        "N": cert.N,
                        "omega": cert.omega,
                        "noise_admissibility": cert.N,
                        "proxy_domain": "geospatial_tabular",
                    },
                )
            gate_result = _GATEKEEPER.evaluate(gate_input)
            gate_decision = gate_result.decision.value
            gear = tau_to_gear(cert.tau).value
        except Exception as e:
            log.warning("Gatekeeper failed for %s/%s: %s", scenario, target, e)

    result = cert.to_dict()
    result.update({
        "scenario": scenario,
        "target": target,
        "task_type": task_type,
        "sigma_inflated": sigma,
        "sigma_inflation": sigma_inflation,
        "gate_decision": gate_decision,
        "gear": gear,
        "used_fallback_cv": used_fallback,
        "fold_scores": spatial_folds,
        "geometry": geom_cell.get("geometry") or {
            "spatial_connectivity": geom_cell.get("spatial_connectivity"),
            "support_coverage": geom_cell.get("support_coverage"),
            "scale_stability": geom_cell.get("scale_stability"),
            "admin_alignment": geom_cell.get("admin_alignment"),
        },
    })

    return result


# ---------------------------------------------------------------------------
# Mode: --collapse
# ---------------------------------------------------------------------------

def run_collapse(s3, upload: bool) -> None:
    """Score-collapse demo: R/S/N simplex vs geometry kappa gradient."""
    log.info("=== COLLAPSE MODE ===")
    geometry_index = _load_geometry_kappa(s3)

    results = []
    for scenario in MODELABLE:
        r0 = _load_r0_results(s3, scenario)
        for target, task_type, transform in TARGETS:
            if scenario not in VALID_CELLS.get(target, []):
                log.info("SKIP %s/%s (not in valid cell matrix)", scenario, target)
                continue
            cert = build_certificate(
                s3, scenario, target, task_type, transform,
                r0, geometry_index,
            )
            results.append(cert)
            log.info(
                "%s / %s: R=%.3f S_sup=%.3f N=%.3f kappa_geom=%s "
                "kappa_compat=%.3f gate=%s",
                scenario, target,
                cert["R"], cert["S_sup"], cert["N"],
                "%.3f" % cert["kappa_geom"] if cert.get("kappa_geom") is not None else "null",
                cert.get("kappa_compat", 0),
                cert.get("gate_decision", "N/A"),
            )

    # Sort by kappa_geom to show collapse gradient
    results.sort(key=lambda c: c.get("kappa_geom") or 0)

    payload = {
        "mode": "collapse",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Score-collapse demo: R/S/N vs geometry kappa gradient",
        "controlplane_available": _HAS_CONTROLPLANE,
        "n_cells": len(results),
        "cells": results,
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_collapse.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Mode: --certificate
# ---------------------------------------------------------------------------

def run_certificate(s3, upload: bool) -> None:
    """Certificate sidecar per (scenario, target)."""
    log.info("=== CERTIFICATE MODE ===")
    geometry_index = _load_geometry_kappa(s3)

    all_certs = {}
    for scenario in MODELABLE:
        r0 = _load_r0_results(s3, scenario)
        scenario_certs = []
        for target, task_type, transform in TARGETS:
            if scenario not in VALID_CELLS.get(target, []):
                log.info("SKIP %s/%s (not in valid cell matrix)", scenario, target)
                continue
            cert = build_certificate(
                s3, scenario, target, task_type, transform,
                r0, geometry_index,
            )
            scenario_certs.append(cert)
            log.info(
                "CERT %s/%s: R=%.3f alpha=%.3f omega=%.3f "
                "kappa_compat=%.3f tau=%.3f gate=%s",
                scenario, target,
                cert["R"], cert["alpha"], cert["omega"],
                cert.get("kappa_compat", 0), cert.get("tau", 0),
                cert.get("gate_decision", "N/A"),
            )
        all_certs[scenario] = scenario_certs

    # Flatten for summary stats
    flat = [c for certs in all_certs.values() for c in certs]

    payload = {
        "mode": "certificate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "RSCT certificate sidecar per (scenario, target)",
        "controlplane_available": _HAS_CONTROLPLANE,
        "scenarios": all_certs,
        "summary": {
            "total_cells": len(flat),
            "gate_decisions": {
                d: sum(1 for c in flat if c.get("gate_decision") == d)
                for d in set(c.get("gate_decision") for c in flat)
            },
            "yrsn_available": flat[0].get("yrsn_available", False) if flat else False,
        },
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_certificates.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Mode: --sweep
# ---------------------------------------------------------------------------

def run_sweep(s3, upload: bool) -> None:
    """Inflation-vs-sigma sweep across all cells."""
    log.info("=== SWEEP MODE ===")
    geometry_index = _load_geometry_kappa(s3)

    inflations = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    sweep_results = []

    for scenario in MODELABLE:
        r0 = _load_r0_results(s3, scenario)
        for target, task_type, transform in TARGETS:
            if scenario not in VALID_CELLS.get(target, []):
                log.info("SKIP %s/%s (not in valid cell matrix)", scenario, target)
                continue
            row = {"scenario": scenario, "target": target, "points": []}
            for inf in inflations:
                cert = build_certificate(
                    s3, scenario, target, task_type, transform,
                    r0, geometry_index,
                    sigma_inflation=inf,
                )
                row["points"].append({
                    "inflation": inf,
                    "sigma": cert.get("sigma"),
                    "sigma_inflated": cert.get("sigma_inflated"),
                    "R": cert["R"],
                    "S_sup": cert["S_sup"],
                    "N": cert["N"],
                    "kappa_compat": cert.get("kappa_compat"),
                    "alpha": cert.get("alpha"),
                    "omega": cert.get("omega"),
                    "tau": cert.get("tau"),
                    "gate_decision": cert.get("gate_decision"),
                    "gear": cert.get("gear"),
                })
            sweep_results.append(row)
            base = row["points"][2]  # inflation=1.0
            log.info(
                "SWEEP %s/%s: base sigma=%.4f, decisions=%s",
                scenario, target,
                base.get("sigma") or 0,
                [p["gate_decision"] for p in row["points"]],
            )

    payload = {
        "mode": "sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Inflation-vs-sigma sweep across cells",
        "controlplane_available": _HAS_CONTROLPLANE,
        "controlplane_config": {
            "kappa_base": _CFG.kappa_base if _CFG else None,
            "sigma_thr": _CFG.sigma_thr if _CFG else None,
            "N_thr": _CFG.N_thr if _CFG else None,
            "alpha_min": _CFG.alpha_min if _CFG else None,
        },
        "inflations": inflations,
        "n_cells": len(sweep_results),
        "cells": sweep_results,
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/rsct_sweep.json",
                           payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSCT certificate sidecar + demos (uses rsct + yrsn_controlplane)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collapse", action="store_true",
                      help="Score-collapse demo")
    mode.add_argument("--certificate", action="store_true",
                      help="Certificate sidecar generator")
    mode.add_argument("--sweep", action="store_true",
                      help="Inflation-vs-sigma sweep")

    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit")
    args = parser.parse_args()

    if args.dry_run:
        n_valid = sum(
            len(scenarios) for scenarios in VALID_CELLS.values()
        )
        log.info("[DRY RUN] Mode: %s",
                 "collapse" if args.collapse else
                 "certificate" if args.certificate else "sweep")
        log.info("[DRY RUN] Scenarios: %s", MODELABLE)
        log.info("[DRY RUN] Targets: %s", [t[0] for t in TARGETS])
        log.info("[DRY RUN] Valid cells: %d (of %d possible)",
                 n_valid, len(MODELABLE) * len(TARGETS))
        for target, scenarios in VALID_CELLS.items():
            log.info("[DRY RUN]   %s: %s", target, scenarios)
        log.info("[DRY RUN] Upload: %s", args.upload)
        log.info("[DRY RUN] controlplane available: %s", _HAS_CONTROLPLANE)
        return

    s3 = get_s3_client()

    if args.collapse:
        run_collapse(s3, args.upload)
    elif args.certificate:
        run_certificate(s3, args.upload)
    elif args.sweep:
        run_sweep(s3, args.upload)

    log.info("Done.")


if __name__ == "__main__":
    main()
