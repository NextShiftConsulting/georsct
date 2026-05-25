"""
validate_spatial.py -- Representation QA for spatial lag and adjacency transforms.

Follows the pattern established in rsct-geocert/data/geocert/v24/run_flood_zones.py:
  1. Critical checks are fatal (abort run)
  2. Non-fatal checks are logged as warnings
  3. All results saved to validation_status.json artifact
  4. Artifact uploaded to S3 for audit trail

Implements Priority 2 (Spatial Lag / W-Matrix QA) from:
  rsct-geocert/V2/METHODS/representation_qa_principle.md
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EXPECTED_CONUS_ZCTAS = 32000  # approximate count for CONUS


def validate_adjacency(
    adjacency: dict,
    df: pd.DataFrame,
    id_col: str = "zcta_id",
    min_zctas: int = None,
) -> dict:
    """Validate W-matrix / adjacency graph against the feature table.

    Returns dict with check results. Raises ValueError on fatal checks.
    """
    checks = {}
    zcta_set = {str(v).zfill(5) for v in df[id_col].values}
    adj_zcta_set = set(adjacency.keys())
    n_data = len(zcta_set)
    n_adj = len(adj_zcta_set)

    # --- Check 1: Row count alignment ---
    in_data_not_adj = zcta_set - adj_zcta_set
    in_adj_not_data = adj_zcta_set - zcta_set
    coverage = 1.0 - len(in_data_not_adj) / max(n_data, 1)

    checks["row_count_alignment"] = {
        "n_data_zctas": n_data,
        "n_adjacency_zctas": n_adj,
        "in_data_not_adjacency": len(in_data_not_adj),
        "in_adjacency_not_data": len(in_adj_not_data),
        "coverage": round(coverage, 4),
    }

    if coverage < 0.90:
        raise ValueError(
            f"FATAL: Adjacency covers only {coverage:.1%} of data ZCTAs "
            f"({len(in_data_not_adj)} missing). Minimum 90% required."
        )
    if coverage < 0.98:
        log.warning("Adjacency coverage %.1f%% -- %d ZCTAs without neighbors",
                     coverage * 100, len(in_data_not_adj))

    log.info("Check 1 (row alignment): PASS -- coverage %.1f%%", coverage * 100)

    # --- Check 2: Self-loops ---
    self_loops = []
    for z, neighbors in adjacency.items():
        if z in neighbors:
            self_loops.append(z)

    checks["self_loops"] = {
        "count": len(self_loops),
        "examples": self_loops[:5],
    }

    if self_loops:
        log.warning("Check 2 (self-loops): %d ZCTAs have self-loops", len(self_loops))
    else:
        log.info("Check 2 (self-loops): PASS -- none found")

    # --- Check 3: Symmetry ---
    asymmetric = 0
    for z, neighbors in adjacency.items():
        for nb in neighbors:
            if nb in adjacency and z not in adjacency[nb]:
                asymmetric += 1

    checks["symmetry"] = {
        "asymmetric_edges": asymmetric,
    }

    if asymmetric > 0:
        log.warning("Check 3 (symmetry): %d asymmetric edges (A->B but not B->A)", asymmetric)
    else:
        log.info("Check 3 (symmetry): PASS -- fully symmetric")

    # --- Check 4: Degree distribution ---
    degrees = [len(nbs) for nbs in adjacency.values()]
    islands = sum(1 for d in degrees if d == 0)
    checks["degree_distribution"] = {
        "min": int(min(degrees)) if degrees else 0,
        "max": int(max(degrees)) if degrees else 0,
        "mean": round(float(np.mean(degrees)), 2) if degrees else 0,
        "median": round(float(np.median(degrees)), 1) if degrees else 0,
        "islands_zero_degree": islands,
    }

    if islands > n_adj * 0.05:
        log.warning("Check 4 (degree): %d island ZCTAs (>5%% of graph)", islands)
    else:
        log.info("Check 4 (degree): PASS -- %d islands, mean degree %.1f",
                 islands, np.mean(degrees) if degrees else 0)

    # --- Check 5: ZCTA count sanity ---
    threshold = min_zctas if min_zctas is not None else int(EXPECTED_CONUS_ZCTAS * 0.5)
    checks["zcta_count_sanity"] = {
        "n_zctas": n_data,
        "expected_conus_approx": EXPECTED_CONUS_ZCTAS,
        "ratio": round(n_data / EXPECTED_CONUS_ZCTAS, 3),
        "min_threshold": threshold,
    }

    if n_data < threshold:
        raise ValueError(
            f"FATAL: Only {n_data} ZCTAs in data -- expected ~{EXPECTED_CONUS_ZCTAS} for CONUS."
        )

    log.info("Check 5 (ZCTA count): PASS -- %d ZCTAs", n_data)

    return checks


def validate_spatial_lags(
    df: pd.DataFrame,
    feature_cols: list,
    adjacency: dict,
    id_col: str = "zcta_id",
) -> dict:
    """Validate spatial lag features after computation.

    Returns dict with check results. Raises ValueError on fatal checks.
    """
    checks = {}
    lag_cols = [f"lag_{c}" for c in feature_cols if f"lag_{c}" in df.columns]
    base_cols = [c for c in feature_cols if f"lag_{c}" in df.columns]

    if not lag_cols:
        raise ValueError("FATAL: No lag columns found in DataFrame after compute_spatial_lags.")

    # --- Check 6: No accidental target leakage in lags ---
    target_cols = [c for c in df.columns if c.startswith("target_")]
    lag_target_overlap = [c for c in lag_cols if c.replace("lag_", "") in
                          [t.replace("target_", "") for t in target_cols]]

    checks["target_leakage"] = {
        "lag_columns": len(lag_cols),
        "target_columns": len(target_cols),
        "suspicious_overlaps": lag_target_overlap,
    }

    if lag_target_overlap:
        raise ValueError(
            f"FATAL: Lag columns overlap with targets: {lag_target_overlap}. "
            "This is target leakage through the spatial graph."
        )

    log.info("Check 6 (target leakage): PASS -- no lag/target overlap")

    # --- Check 7: Lag vs base distribution comparison ---
    corr_stats = []
    for base_c, lag_c in zip(base_cols, lag_cols):
        base_vals = df[base_c].values
        lag_vals = df[lag_c].values
        valid = ~(np.isnan(base_vals) | np.isnan(lag_vals))
        if valid.sum() > 100:
            r = float(np.corrcoef(base_vals[valid], lag_vals[valid])[0, 1])
            corr_stats.append({"feature": base_c, "base_lag_corr": round(r, 4)})

    if corr_stats:
        corrs = [s["base_lag_corr"] for s in corr_stats]
        mean_corr = float(np.mean(corrs))
        low_corr = [s for s in corr_stats if abs(s["base_lag_corr"]) < 0.1]
    else:
        mean_corr = 0.0
        low_corr = []

    checks["lag_base_correlation"] = {
        "mean_correlation": round(mean_corr, 4),
        "n_features_checked": len(corr_stats),
        "n_low_correlation": len(low_corr),
        "low_correlation_features": low_corr[:5],
    }

    if mean_corr < 0.2:
        log.warning("Check 7 (lag correlation): mean base-lag r=%.3f -- "
                     "spatial lags may not carry meaningful signal", mean_corr)
    else:
        log.info("Check 7 (lag correlation): PASS -- mean base-lag r=%.3f", mean_corr)

    # --- Check 8: Zero-lag rows (islands) ---
    lag_matrix = df[lag_cols].values
    zero_rows = int(np.all(lag_matrix == 0, axis=1).sum())
    zero_pct = zero_rows / max(len(df), 1)

    checks["zero_lag_rows"] = {
        "count": zero_rows,
        "pct": round(zero_pct, 4),
    }

    if zero_pct > 0.10:
        log.warning("Check 8 (zero-lag rows): %d rows (%.1f%%) have all-zero lags",
                     zero_rows, zero_pct * 100)
    else:
        log.info("Check 8 (zero-lag rows): PASS -- %d rows (%.1f%%)", zero_rows, zero_pct * 100)

    # --- Check 9: NaN/Inf in lag features ---
    n_nan = int(np.isnan(lag_matrix).sum())
    n_inf = int(np.isinf(lag_matrix).sum())

    checks["lag_nan_inf"] = {"nan_count": n_nan, "inf_count": n_inf}

    if n_inf > 0:
        raise ValueError(f"FATAL: {n_inf} Inf values in lag features.")
    if n_nan > 0:
        log.warning("Check 9 (NaN/Inf): %d NaN values in lag features", n_nan)
    else:
        log.info("Check 9 (NaN/Inf): PASS")

    return checks


def run_spatial_validation(
    df: pd.DataFrame,
    feature_cols: list,
    adjacency: dict,
    output_dir: str = None,
    id_col: str = "zcta_id",
    min_zctas: int = None,
) -> bool:
    """Run full spatial representation validation. Returns True if all critical checks pass.

    Saves validation_status.json artifact if output_dir is provided.
    """
    log.info("=== SPATIAL REPRESENTATION VALIDATION ===")
    log.info("Checks: adjacency alignment, self-loops, symmetry, degree,")
    log.info("        target leakage, lag correlation, zero-lags, NaN/Inf")

    adj_checks = validate_adjacency(adjacency, df, id_col, min_zctas=min_zctas)
    lag_checks = validate_spatial_lags(df, feature_cols, adjacency, id_col)

    all_checks = {**adj_checks, **lag_checks}
    status = "PASS"
    log.info("SPATIAL VALIDATION: %s", status)

    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_type": "spatial_lag_w_matrix",
        "status": status,
        "n_zctas": len(df),
        "n_features": len(feature_cols),
        "n_lag_features": sum(1 for c in df.columns if c.startswith("lag_")),
        "n_adjacency_zctas": len(adjacency),
        "checks": all_checks,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifact_path = out / "spatial_validation_status.json"
        artifact_path.write_text(json.dumps(artifact, indent=2))
        log.info("Saved: %s", artifact_path)

    return True
