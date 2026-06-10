"""
compute_spatial_sidecar.py -- DOE Spatial Diagnostics (Non-Locking Sidecar)

Sections 1, 2, 4 from DOE_spatial_diagnostics.md:
  S1: LISA residual clusters with BH-FDR + cluster attenuation (Phase 4d)
  S2: GWR/MGWR non-stationarity probe (Phase 0.6)
  S4: Geary's C companion (Phase 4d)

All outputs go to results/s035/sidecar/ -- never touches the locked pipeline.

Section 3 (regionalization robustness) is in compute_spatial_sidecar_regionalize.py
because it re-runs training via the existing train_r0/r1 entry points.

Architecture: the 11 functions below are PURE (no I/O, no S3, no logging side
effects). main() does all I/O. This keeps them unit-testable against fixtures,
which matters for a pre-registration-heavy paper where apply_fdr_bh,
compute_deviance_residuals, and clopper_pearson_ci are precisely the functions
easiest to get subtly wrong.

Usage:
    python compute_spatial_sidecar.py --section lisa   --upload
    python compute_spatial_sidecar.py --section gwr    --upload
    python compute_spatial_sidecar.py --section geary  --upload
    python compute_spatial_sidecar.py --section all    --upload
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
from _coverage_common import BUCKET, get_s3_client, load_adjacency, level_prefix
from _s3_result import upload_json_result
from compute_residual_lisa import (
    build_weights_from_adjacency,
    MODELABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

SIDECAR_PREFIX = "results/s035/sidecar"
RESULTS_PREFIX = "results/s035"
LEVELS = ["r0", "r1", "r2"]
TARGETS = ["obs_nfip_event_claims", "obs_has_311", "obs_has_hwm"]
BINARY_TARGETS = {"obs_has_311", "obs_has_hwm"}

# Pre-registered GWR feature subset (frozen -- do not change after launch)
GWR_FEATURES = [
    "flood_pct_zone_a",
    "twi_acc_twi",
    "slope_basin_slope",
    "acs_median_hh_income",
    "svi_overall",
    "population",
]
GWR_MIN_N = 50
GWR_SEED = 42
GWR_CRITERION = "AICc"        # Frozen for pre-registration
NONSTAT_THRESHOLD = 0.5       # Frozen: coefficient relative spread cutoff
LISA_SEED = 42
LISA_PERMUTATIONS = 999
GEARY_PERMUTATIONS = 999


# ═══════════════════════════════════════════════════════════════════════════
# PURE FUNCTIONS (no I/O, no S3, no logging -- unit-testable)
# ═══════════════════════════════════════════════════════════════════════════


# --- #1: BH-FDR correction ---

def apply_fdr_bh(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction on LISA pseudo p-values.

    Returns adjusted p-values. A ZCTA is significant if adjusted_p < q.

    Note: BH assumes positive dependence. LISA statistics are spatially
    correlated by construction, so BH is approximate here. Acceptable for
    descriptive use; cite Caldas de Castro & Singer (2006) if challenged.
    """
    from statsmodels.stats.multitest import multipletests

    _, pvals_adj, _, _ = multipletests(p_values, alpha=q, method="fdr_bh")
    return pvals_adj


# --- #2: Deviance residuals for binary targets ---

def compute_deviance_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Deviance residuals for binary classification.

    d_i = sign(y_i - p_i) * sqrt(-2 * [y_i*log(p_i) + (1-y_i)*log(1-p_i)])

    Clamps predicted probabilities to [1e-7, 1-1e-7] to avoid inf/NaN
    on confident predictions.
    """
    p = np.clip(y_pred.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y = y_true.astype(np.float64)
    raw = -2.0 * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return np.sign(y - p) * np.sqrt(np.maximum(raw, 0.0))


# --- #3: LISA with FDR ---

def compute_lisa_fdr(
    residuals: pd.Series,
    w: "libpysal.weights.W",
    permutations: int = LISA_PERMUTATIONS,
    fdr_q: float = 0.05,
    seed: int = LISA_SEED,
) -> pd.DataFrame:
    """Local Moran's I with Benjamini-Hochberg FDR correction.

    Enhanced version of compute_residual_lisa.compute_lisa() -- uses BH-FDR
    instead of raw alpha cutoff, per DOE spec Section 1.

    Returns DataFrame indexed by zcta_id:
        local_moran_I, local_moran_p_raw, local_moran_p_fdr, cluster_label
    """
    from esda.moran import Moran_Local

    aligned = residuals.reindex(w.id_order)
    y = aligned.values.astype(np.float64)

    nan_mask = np.isnan(y)
    if nan_mask.any():
        y = y.copy()
        y[nan_mask] = 0.0

    np.random.seed(seed)
    lisa = Moran_Local(y, w, permutations=permutations)

    # BH-FDR correction
    p_adj = apply_fdr_bh(lisa.p_sim, q=fdr_q)

    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    clusters = []
    for i in range(len(y)):
        if p_adj[i] >= fdr_q:
            clusters.append("ns")
        else:
            clusters.append(quad_labels.get(lisa.q[i], "ns"))

    return pd.DataFrame({
        "local_moran_I": lisa.Is,
        "local_moran_p_raw": lisa.p_sim,
        "local_moran_p_fdr": p_adj,
        "cluster_label": clusters,
    }, index=w.id_order)


# --- #4: LISA cell-level rollup ---

def compute_lisa_rollup(
    lisa_df: pd.DataFrame,
    global_moran_i: float,
) -> dict:
    """Aggregate per-ZCTA LISA to cell-level summary.

    Only counts cluster labels on FDR-significant ZCTAs. An HH label on
    a non-significant ZCTA does not count toward frac_HH.
    """
    n = len(lisa_df)
    if n == 0:
        return {
            "frac_HH": 0.0, "frac_LL": 0.0, "frac_outlier": 0.0,
            "frac_significant": 0.0, "global_moran_I": global_moran_i,
            "n_zctas": 0,
        }

    labels = lisa_df["cluster_label"]
    # cluster_label is already "ns" for non-significant ZCTAs,
    # so only HH/LL/HL/LH labels are FDR-significant by construction.
    return {
        "frac_HH": float((labels == "HH").sum() / n),
        "frac_LL": float((labels == "LL").sum() / n),
        "frac_outlier": float(labels.isin(["HL", "LH"]).sum() / n),
        "frac_significant": float((labels != "ns").sum() / n),
        "global_moran_I": global_moran_i,
        "n_zctas": n,
    }


# --- #5: Clopper-Pearson binomial CI ---

def clopper_pearson_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Returns (lower, upper) bounds of the 1-alpha CI.
    """
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1.0 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lower), float(upper))


# --- #6: Cluster attenuation metric ---

def compute_cluster_attenuation(
    rollups: dict[str, dict],
) -> dict:
    """Compute lisa_cluster_frac deltas across representation levels.

    Args:
        rollups: Dict keyed by level ("r0","r1","r2"), values from compute_lisa_rollup().
    """
    def cluster_frac(level: str) -> float:
        r = rollups.get(level, {})
        return r.get("frac_HH", 0.0) + r.get("frac_LL", 0.0)

    frac_r0 = cluster_frac("r0")
    frac_r1 = cluster_frac("r1")
    frac_r2 = cluster_frac("r2")

    delta_r0_r1 = frac_r1 - frac_r0
    delta_r1_r2 = frac_r2 - frac_r1

    return {
        "lisa_cluster_frac_r0": frac_r0,
        "lisa_cluster_frac_r1": frac_r1,
        "lisa_cluster_frac_r2": frac_r2,
        "delta_R0_R1": delta_r0_r1,
        "delta_R0_R1_sign": "attenuated" if delta_r0_r1 < 0 else "persisted",
        "delta_R1_R2": delta_r1_r2,
        "delta_R1_R2_sign": "attenuated" if delta_r1_r2 < 0 else "persisted",
    }


def compute_attenuation_summary(
    cell_attenuations: list[dict],
) -> dict:
    """Cross-cell summary: fraction attenuated + Clopper-Pearson CI.

    Labeled decorative at n<=9 per DOE spec.
    """
    n = len(cell_attenuations)
    k = sum(1 for c in cell_attenuations if c.get("delta_R0_R1_sign") == "attenuated")
    lo, hi = clopper_pearson_ci(k, n)

    return {
        "n_cells": n,
        "n_attenuated_R0_R1": k,
        "frac_attenuated": k / n if n > 0 else 0.0,
        "clopper_pearson_95_ci": [lo, hi],
        "note": f"Decorative binomial CI at n={n}. Not a hypothesis test.",
    }


# --- #7: Geary's C ---

def compute_geary(
    values: np.ndarray,
    w: "libpysal.weights.W",
    permutations: int = GEARY_PERMUTATIONS,
    seed: int = LISA_SEED,
) -> dict:
    """Geary's C on residuals.

    C ~ 1: no autocorrelation. C < 1: positive. C > 1: negative.
    Companion to Moran's I -- more sensitive to local/short-range dissimilarity.
    """
    from esda.geary import Geary

    np.random.seed(seed)
    g = Geary(values, w, permutations=permutations)

    if g.C < 0.95:
        interp = "positive_autocorrelation"
    elif g.C > 1.05:
        interp = "negative_autocorrelation"
    else:
        interp = "no_significant_autocorrelation"

    return {
        "geary_C": float(g.C),
        "geary_p": float(g.p_sim),
        "interpretation": interp,
    }


# --- #8: GWR non-stationarity probe ---

def fit_gwr_probe(
    coords_5070: list[tuple[float, float]],
    y: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
) -> dict:
    """GWR non-stationarity probe on the full cross-section.

    PURE FUNCTION: takes pre-projected EPSG:5070 coordinates, z-scored y and X.
    Caller is responsible for projection and standardization.

    Solver-independent structural description -- NOT predictive performance.
    Uses adaptive bisquare kernel, AICc bandwidth selection (deterministic).

    For gwr_frac_nonstationary: uses descriptive coefficient-spread index
    (std / |mean| per coefficient across locations), NOT mgwr's
    parameter-stationarity test, which is version-fragile.
    """
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    import spreg

    n_obs = len(y)

    # Global OLS baseline (for AICc comparison)
    # spreg.OLS reports AIC, not AICc.  Apply small-sample correction so
    # delta_aicc compares like with like (GWR reports AICc).
    y_col = y.reshape(-1, 1)
    k_params = X.shape[1] + 1  # intercept
    ols = spreg.OLS(y_col, X, name_y="target", name_x=feature_names)
    if hasattr(ols, "aic"):
        ols_aic = float(ols.aic)
        correction = (2.0 * k_params * (k_params + 1)) / max(n_obs - k_params - 1, 1)
        ols_aicc = ols_aic + correction
    else:
        ols_aicc = None

    # GWR: adaptive bisquare, AICc bandwidth selection
    np.random.seed(GWR_SEED)
    selector = Sel_BW(coords_5070, y_col, X, spherical=False)  # projected coords
    bw = selector.search(bw_min=2, criterion=GWR_CRITERION)

    gwr_result = GWR(coords_5070, y_col, X, bw).fit()
    gwr_aicc = float(gwr_result.aicc)

    # Local R^2
    local_r2 = gwr_result.localR2.flatten()
    lr2_mean = float(np.mean(local_r2))
    local_r2_cv = float(np.std(local_r2) / lr2_mean) if lr2_mean > 1e-12 else 0.0

    # Non-stationarity: descriptive coefficient-spread index
    # std(local_coef) / |mean(local_coef)| per feature, across locations.
    # Avoids mgwr's parameter-stationarity test (version-fragile).
    coef_std = np.std(gwr_result.params, axis=0)
    coef_mean_abs = np.abs(np.mean(gwr_result.params, axis=0))
    coef_mean_abs[coef_mean_abs < 1e-12] = 1.0
    norm_coef_std = coef_std / coef_mean_abs

    # Fraction with relative variation exceeding frozen threshold
    frac_nonstationary = float(np.mean(norm_coef_std > NONSTAT_THRESHOLD))
    # Mean normalized spread as summary scalar
    nonstationarity = float(np.mean(norm_coef_std))

    delta_aicc = (ols_aicc - gwr_aicc) if ols_aicc is not None else None

    return {
        "status": "COMPLETE",
        "n_zcta": n_obs,
        "bandwidth": float(bw),
        "ols_aicc": ols_aicc,
        "gwr_aicc": gwr_aicc,
        "gwr_delta_aicc": delta_aicc,
        "gwr_local_r2_mean": lr2_mean,
        "gwr_local_r2_cv": local_r2_cv,
        "gwr_frac_nonstationary": frac_nonstationary,
        "gwr_nonstationarity": nonstationarity,
        "features_used": feature_names,
        "per_feature_nonstationarity": {
            f: float(s) for f, s in zip(feature_names, norm_coef_std)
        },
        "local_r2": local_r2.tolist(),
    }


# --- #9: MGWR probe (optional, primary cell only) ---

def fit_mgwr_probe(
    coords_5070: list[tuple[float, float]],
    y: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Multi-scale GWR -- per-covariate bandwidths.

    Optional, run on primary cell only. Reports which features are global vs local.
    Falls back to GWR-only result if MGWR fails to converge (common at small n).

    PURE FUNCTION: takes pre-projected EPSG:5070 coords, z-scored y and X.
    """
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW

    y_col = y.reshape(-1, 1)

    np.random.seed(GWR_SEED)
    try:
        selector = Sel_BW(coords_5070, y_col, X, multi=True, spherical=False)
        selector.search(multi_bw_min=[4])
        mgwr_result = MGWR(coords_5070, y_col, X, selector, sigma2_v1=True).fit()

        per_var_bw = {f: float(b) for f, b in zip(feature_names, selector.bw_)}

        return {
            "status": "COMPLETE",
            "n_zcta": len(y),
            "per_variable_bandwidth": per_var_bw,
            "mgwr_aicc": float(mgwr_result.aicc),
            "features_used": feature_names,
        }
    except Exception as exc:
        return {
            "status": "MGWR_CONVERGENCE_FAILURE",
            "n_zcta": len(y),
            "error": str(exc),
            "note": "MGWR failed to converge; GWR result is primary.",
        }


# --- #10: Triangulation table (kappa_geom vs GWR vs uplift) ---

def compute_triangulation_table(
    kappa_geom_cells: list[dict],
    gwr_cells: list[dict],
    uplift_cells: list[dict],
) -> dict:
    """Rank table: kappa_geom vs gwr_nonstationarity vs observed R0->R1 uplift.

    Reports Spearman rho with bootstrap 95% CI, explicitly labeled as observed
    triangulation, NOT a hypothesis test, NOT in the Holm-Bonferroni family.
    """
    from scipy.stats import spearmanr

    # Filter all inputs to primary target only. Without this, uplift_map
    # silently keeps whichever target appears last per scenario.
    primary = "obs_nfip_event_claims"
    kappa_filt = [c for c in kappa_geom_cells if c.get("target", primary) == primary]
    gwr_filt = [c for c in gwr_cells if c.get("target", primary) == primary]
    uplift_filt = [c for c in uplift_cells if c.get("target", primary) == primary]

    kappa_map = {c["scenario"]: c.get("kappa_geom", np.nan) for c in kappa_filt}
    kappa_prior_map = {c["scenario"]: c.get("kappa_prior", c.get("kappa_geom", np.nan)) for c in kappa_filt}
    gwr_map = {c["scenario"]: c.get("gwr_nonstationarity", np.nan) for c in gwr_filt}
    uplift_map = {c["scenario"]: c.get("uplift_r0_r1", np.nan) for c in uplift_filt}

    scenarios = sorted(set(kappa_map) & set(gwr_map) & set(uplift_map))
    rows = []
    for s in scenarios:
        kg, gn, up = kappa_map[s], gwr_map[s], uplift_map[s]
        if not any(np.isnan([kg, gn, up])):
            rows.append({"scenario": s, "kappa_geom": kg,
                         "kappa_prior": kappa_prior_map.get(s, kg),  # Q-007 alias
                         "gwr_nonstationarity": gn, "uplift_r0_r1": up})

    if len(rows) < 3:
        return {"status": "INSUFFICIENT_CELLS", "n_cells": len(rows)}

    df = pd.DataFrame(rows)
    for col in ["kappa_geom", "gwr_nonstationarity", "uplift_r0_r1"]:
        df[f"{col}_rank"] = df[col].rank()

    def _spearman_boot(a, b, n_boot=1000):
        rho, _ = spearmanr(a, b)
        rng = np.random.default_rng(42)
        rhos = []
        for _ in range(n_boot):
            idx = rng.choice(len(a), size=len(a), replace=True)
            r, _ = spearmanr(a[idx], b[idx])
            if not np.isnan(r):
                rhos.append(r)
        lo = float(np.percentile(rhos, 2.5)) if rhos else np.nan
        hi = float(np.percentile(rhos, 97.5)) if rhos else np.nan
        return float(rho), lo, hi

    kg_a = df["kappa_geom"].values
    gn_a = df["gwr_nonstationarity"].values
    up_a = df["uplift_r0_r1"].values

    return {
        "status": "COMPLETE",
        "n_cells": len(rows),
        "rank_table": df.to_dict(orient="records"),
        "spearman": {
            "kappa_geom_vs_uplift": dict(zip(
                ["rho", "ci_lo", "ci_hi"], _spearman_boot(kg_a, up_a))),
            "kappa_prior_vs_uplift": dict(zip(
                ["rho", "ci_lo", "ci_hi"], _spearman_boot(kg_a, up_a))),  # Q-007 alias (same values)
            "gwr_vs_uplift": dict(zip(
                ["rho", "ci_lo", "ci_hi"], _spearman_boot(gn_a, up_a))),
            "kappa_geom_vs_gwr": dict(zip(
                ["rho", "ci_lo", "ci_hi"], _spearman_boot(kg_a, gn_a))),
            "kappa_prior_vs_gwr": dict(zip(
                ["rho", "ci_lo", "ci_hi"], _spearman_boot(kg_a, gn_a))),  # Q-007 alias (same values)
        },
        "note": "Observed triangulation. NOT a hypothesis test. NOT in Holm-Bonferroni family.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# I/O + ORCHESTRATION (all S3 access lives here, never in pure functions)
# ═══════════════════════════════════════════════════════════════════════════


def _load_predictions(s3, level: str, scenario: str) -> pd.DataFrame:
    key = f"{RESULTS_PREFIX}/{level_prefix(level)}_{scenario}_predictions.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded %d predictions from %s", len(df), key)
        return df
    except Exception:
        log.warning("Predictions not found: %s", key)
        return pd.DataFrame()


def _load_processed_parquet(s3, scenario: str) -> pd.DataFrame:
    from _coverage_common import load_processed_parquet
    return load_processed_parquet(s3, scenario)


def _load_kappa_geom(s3) -> list[dict]:
    key = f"{RESULTS_PREFIX}/geometry_kappa.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read()).get("cells", [])
    except Exception:
        log.warning("geometry_kappa.json not found")
        return []


def _load_uplift(s3) -> list[dict]:
    key = f"{RESULTS_PREFIX}/uplift_table.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return data.get("cells", data.get("rows", []))
    except Exception:
        log.warning("uplift_table.json not found")
        return []


def _compute_zcta_residuals(
    preds: pd.DataFrame, target: str,
) -> pd.Series:
    """Compute mean per-ZCTA residuals from predictions DataFrame."""
    mask = preds["target"] == target
    if "solver" in preds.columns:
        mask = mask & (preds["solver"] == "histgbdt")
    cell = preds[mask].copy()

    if cell.empty or "y_true" not in cell.columns:
        return pd.Series(dtype=np.float64)

    cell["zcta_id"] = cell["zcta_id"].astype(str)
    y_true = cell["y_true"].values
    y_pred = cell["y_pred"].values

    if target in BINARY_TARGETS:
        resid = compute_deviance_residuals(y_true, y_pred)
    else:
        resid = y_true - y_pred

    cell["residual"] = resid
    return cell.groupby("zcta_id")["residual"].mean()


# --- Section 1: LISA ---

def run_section_lisa(s3, adj_df: pd.DataFrame, upload: bool) -> dict:
    """Section 1: LISA residual clusters with BH-FDR + cluster attenuation."""
    log.info("=" * 60)
    log.info("  SECTION 1: LISA RESIDUAL CLUSTERS (Phase 4d)")
    log.info("=" * 60)

    all_rollups = []
    all_attenuations = []

    for scenario in MODELABLE:
        for target in TARGETS:
            level_rollups = {}

            for level in LEVELS:
                preds = _load_predictions(s3, level, scenario)
                if preds.empty:
                    continue

                zcta_resid = _compute_zcta_residuals(preds, target)
                if zcta_resid.empty:
                    continue

                zcta_ids = sorted(zcta_resid.index.tolist())
                w = build_weights_from_adjacency(adj_df, zcta_ids)
                if w.n < 10:
                    log.warning("Skipping %s/%s/%s: %d ZCTAs", scenario, target, level, w.n)
                    continue

                # LISA with FDR (pure function)
                lisa_df = compute_lisa_fdr(zcta_resid, w)

                # Global Moran's I
                from esda.moran import Moran
                np.random.seed(LISA_SEED)
                values = zcta_resid.reindex(w.id_order).fillna(0).values
                global_m = Moran(values, w, permutations=LISA_PERMUTATIONS)

                # Rollup (pure function)
                rollup = compute_lisa_rollup(lisa_df, float(global_m.I))
                rollup.update({"scenario": scenario, "target": target, "level": level})
                level_rollups[level] = rollup
                all_rollups.append(rollup)

                log.info(
                    "  %s/%s/%s: HH=%.3f LL=%.3f sig=%.3f I=%.4f",
                    scenario, target, level,
                    rollup["frac_HH"], rollup["frac_LL"],
                    rollup["frac_significant"], rollup["global_moran_I"],
                )

                if upload:
                    lisa_df = lisa_df.copy()
                    lisa_df.index.name = "zcta_id"
                    lisa_df["residual"] = zcta_resid.reindex(lisa_df.index)
                    buf = io.BytesIO()
                    lisa_df.reset_index().to_parquet(buf, compression="zstd")
                    skey = f"{SIDECAR_PREFIX}/lisa_{scenario}_{target}_{level}.parquet"
                    s3.put_object(Bucket=BUCKET, Key=skey, Body=buf.getvalue())
                    log.info("Uploaded %s", skey)

            if "r0" in level_rollups:
                att = compute_cluster_attenuation(level_rollups)
                att.update({"scenario": scenario, "target": target})
                all_attenuations.append(att)

    attenuation_summary = compute_attenuation_summary(all_attenuations)

    payload = {
        "section": "lisa",
        "phase": "4d",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rollups": all_rollups,
        "attenuations": all_attenuations,
        "attenuation_summary": attenuation_summary,
    }

    if upload:
        if all_rollups:
            buf = io.BytesIO()
            pd.DataFrame(all_rollups).to_parquet(buf, compression="zstd")
            s3.put_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/lisa_rollup.parquet",
                          Body=buf.getvalue())
        upload_json_result(s3, BUCKET, f"{SIDECAR_PREFIX}/lisa_results.json", payload)

    return payload


# --- Section 2: GWR ---

def run_section_gwr(s3, upload: bool) -> dict:
    """Section 2: GWR/MGWR non-stationarity probe."""
    log.info("=" * 60)
    log.info("  SECTION 2: GWR NON-STATIONARITY PROBE (Phase 0.6)")
    log.info("=" * 60)

    import geopandas as gpd

    primary_target = "obs_nfip_event_claims"
    gwr_cells = []

    for scenario in MODELABLE:
        log.info("GWR probe: %s", scenario)

        try:
            df = _load_processed_parquet(s3, scenario)
        except Exception as e:
            log.warning("Cannot load %s: %s", scenario, e)
            continue

        df["zcta_id"] = df["zcta_id"].astype(str)

        # --- Geometry: MUST be EPSG:5070 (equal-area projected) ---
        # mgwr computes kernel distances in coordinate units. If you feed it
        # lat/lon, bandwidths and local R^2 are geometrically meaningless,
        # and it fails SILENTLY (no crash, just wrong numbers).
        gdf = None

        # Try pre-projected ZCTA boundaries
        for zcta_key in [
            "raw/geocertdb2026/zcta_boundaries_5070.parquet",
            "raw/geocertdb2026/zcta_boundaries.parquet",
            "raw/geocertdb2026/zcta5_boundaries.parquet",
        ]:
            try:
                obj = s3.get_object(Bucket=BUCKET, Key=zcta_key)
                zcta_gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
                if "zcta_id" in zcta_gdf.columns:
                    zcta_gdf["zcta_id"] = zcta_gdf["zcta_id"].astype(str)
                # Project to EPSG:5070 if needed
                if zcta_gdf.crs is None or zcta_gdf.crs.to_epsg() != 5070:
                    zcta_gdf = zcta_gdf.to_crs("EPSG:5070")
                merged = zcta_gdf[["zcta_id", "geometry"]].merge(df, on="zcta_id")
                gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:5070")
                break
            except Exception:
                continue

        # Fallback: lat/lon -> point geometry -> project
        if gdf is None and "latitude" in df.columns and "longitude" in df.columns:
            from shapely.geometry import Point
            geom = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
            gdf = gdf.to_crs("EPSG:5070")
            log.warning("Using lat/lon point centroids projected to EPSG:5070 for %s", scenario)

        if gdf is None:
            log.warning("No geometry for %s, skipping GWR", scenario)
            gwr_cells.append({"scenario": scenario, "status": "NO_GEOMETRY"})
            continue

        # Filter to available features + complete cases
        available = [f for f in GWR_FEATURES if f in gdf.columns]
        if len(available) < 3:
            gwr_cells.append({"scenario": scenario, "status": "INSUFFICIENT_FEATURES",
                              "n_available": len(available)})
            continue

        sub = gdf[["zcta_id", primary_target] + available + ["geometry"]].dropna()
        if len(sub) < GWR_MIN_N:
            gwr_cells.append({"scenario": scenario, "status": "INSUFFICIENT_N",
                              "n_zcta": len(sub)})
            continue

        # Z-score (pure data prep before handing to pure function)
        y_raw = sub[primary_target].values.astype(np.float64)
        X_raw = sub[available].values.astype(np.float64)

        y_std = y_raw.std()
        X_std = X_raw.std(axis=0)
        X_std[X_std < 1e-12] = 1.0
        if y_std < 1e-12:
            gwr_cells.append({"scenario": scenario, "status": "ZERO_VARIANCE_TARGET"})
            continue

        y = (y_raw - y_raw.mean()) / y_std
        X = (X_raw - X_raw.mean(axis=0)) / X_std

        # EPSG:5070 centroids
        centroids = sub.geometry.centroid
        coords = list(zip(centroids.x, centroids.y))

        # Pure function call
        result = fit_gwr_probe(coords, y, X, available)
        result["scenario"] = scenario
        result["target"] = primary_target

        # MGWR on primary cell only (with try/except fallback)
        if scenario == "houston" and result.get("status") == "COMPLETE":
            log.info("Running MGWR on houston (primary cell)")
            result["mgwr"] = fit_mgwr_probe(coords, y, X, available)

        # Save local-R^2 to S3 for figure rendering
        if upload and result.get("status") == "COMPLETE":
            zcta_ids = sub["zcta_id"].tolist() if "zcta_id" in sub.columns else []
            lr2 = result.pop("local_r2", [])
            if zcta_ids and lr2:
                buf = io.BytesIO()
                pd.DataFrame({"zcta_id": zcta_ids, "local_r2": lr2}).to_parquet(
                    buf, compression="zstd")
                s3.put_object(
                    Bucket=BUCKET,
                    Key=f"{SIDECAR_PREFIX}/gwr_local_r2_{scenario}.parquet",
                    Body=buf.getvalue(),
                )
                log.info("Uploaded gwr_local_r2_%s.parquet", scenario)
        else:
            result.pop("local_r2", None)

        gwr_cells.append(result)

    # Triangulation (if kappa_geom and uplift exist)
    kappa_cells = _load_kappa_geom(s3)
    uplift_cells = _load_uplift(s3)
    triangulation = compute_triangulation_table(
        [c for c in kappa_cells if c.get("target") == primary_target],
        [c for c in gwr_cells if c.get("status") == "COMPLETE"],
        uplift_cells,
    )

    payload = {
        "section": "gwr",
        "phase": "0.6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gwr_features_frozen": GWR_FEATURES,
        "projection": "EPSG:5070",
        "cells": gwr_cells,
        "triangulation": triangulation,
    }

    if upload:
        upload_json_result(s3, BUCKET, f"{SIDECAR_PREFIX}/gwr_nonstationarity.json", payload)

    return payload


# --- Section 4: Geary's C ---

def run_section_geary(s3, adj_df: pd.DataFrame, upload: bool) -> dict:
    """Section 4: Geary's C on residuals -- companion to Moran's I."""
    log.info("=" * 60)
    log.info("  SECTION 4: GEARY'S C COMPANION (Phase 4d)")
    log.info("=" * 60)

    all_geary = []

    for scenario in MODELABLE:
        for target in TARGETS:
            for level in LEVELS:
                preds = _load_predictions(s3, level, scenario)
                if preds.empty:
                    continue

                zcta_resid = _compute_zcta_residuals(preds, target)
                if zcta_resid.empty:
                    continue

                zcta_ids = sorted(zcta_resid.index.tolist())
                w = build_weights_from_adjacency(adj_df, zcta_ids)
                if w.n < 10:
                    continue

                values = zcta_resid.reindex(w.id_order).fillna(0).values.astype(np.float64)

                # Pure function call
                result = compute_geary(values, w)
                result.update({"scenario": scenario, "target": target, "level": level})
                all_geary.append(result)

                log.info(
                    "  %s/%s/%s: C=%.4f (p=%.4f) %s",
                    scenario, target, level,
                    result["geary_C"], result["geary_p"], result["interpretation"],
                )

    # Compute deltas
    geary_deltas = []
    for scenario in MODELABLE:
        for target in TARGETS:
            by_level = {g["level"]: g for g in all_geary
                        if g["scenario"] == scenario and g["target"] == target}
            if "r0" in by_level and "r1" in by_level:
                geary_deltas.append({
                    "scenario": scenario,
                    "target": target,
                    "geary_C_r0": by_level["r0"]["geary_C"],
                    "geary_C_r1": by_level["r1"]["geary_C"],
                    "geary_delta_R0_R1": by_level["r1"]["geary_C"] - by_level["r0"]["geary_C"],
                })

    payload = {
        "section": "geary",
        "phase": "4d",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": all_geary,
        "deltas": geary_deltas,
    }

    if upload:
        if all_geary:
            buf = io.BytesIO()
            pd.DataFrame(all_geary).to_parquet(buf, compression="zstd")
            s3.put_object(Bucket=BUCKET, Key=f"{SIDECAR_PREFIX}/geary_rollup.parquet",
                          Body=buf.getvalue())
        upload_json_result(s3, BUCKET, f"{SIDECAR_PREFIX}/geary_results.json", payload)

    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOE Spatial Diagnostics Sidecar (Sections 1, 2, 4)")
    parser.add_argument("--section", required=True,
                        choices=["lisa", "gwr", "geary", "all"])
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    adj_df = None
    if args.section in ("lisa", "geary", "all"):
        try:
            adj_df = load_adjacency(s3)
        except FileNotFoundError:
            log.error("ZCTA adjacency not found. Required for LISA/Geary.")
            sys.exit(1)

    results = {}

    if args.section in ("lisa", "all"):
        results["lisa"] = run_section_lisa(s3, adj_df, args.upload)

    if args.section in ("gwr", "all"):
        results["gwr"] = run_section_gwr(s3, args.upload)

    if args.section in ("geary", "all"):
        results["geary"] = run_section_geary(s3, adj_df, args.upload)

    print("\n" + "=" * 60)
    print("  SPATIAL SIDECAR SUMMARY")
    print("=" * 60)
    for section, payload in results.items():
        print(f"\n--- {section.upper()} ---")
        # Truncate for terminal readability
        summary = {k: v for k, v in payload.items()
                   if k not in ("rollups", "cells", "attenuations", "deltas")}
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
