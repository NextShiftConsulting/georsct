#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   author: Martin
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""compute_verdicts.py -- Phase 8: Six-geometry sufficiency verdicts.

Reads existing mmar-input data (diagnostics, LISA, money table, certificates)
and prediction parquets from S3 to produce per-geometry verdict JSONs for the
paper's Table 1.

Geometries:
  1. Prediction  -- already computed in compute_uplift_table.py (H2 Wilcoxon)
  2. Ranking     -- Kendall tau-b(R0, R1) + fidelity-delta vs FAST
  3. Clustering  -- Moran's I significance after spatial ladder level
  4. Transfer    -- LEO retention ratio from diagnostics
  5. Relational  -- W-matrix ablation (full vs no-wlag vs wlag-only)
  6. Allocation  -- R2 temporal coverage by event vintage

Outputs:
  results/s035/verdicts.json  (all six geometries)
  Local copy to exp/s035-model-ladder/mmar-input/verdicts.json

Usage:
    python compute_verdicts.py --upload
    python compute_verdicts.py --dry-run
"""

import argparse
import io
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, level_prefix
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SIDECAR_PREFIX = "results/s035/sidecar"
LOCAL_MMAR_DIR = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "mmar-input"
PRIMARY_TARGET = "obs_nfip_event_claims"
PRIMARY_SOLVER = "histgbdt"

# Pre-committed thresholds from DOE / appendix
TAU_B_REORDER_THRESHOLD = 0.90  # below this = meaningful reorder
TRANSFER_RETENTION_THRESHOLD = 0.50  # pre-committed fraction for SUPPORTED

# Event vintage boundaries for allocation verdict
# MRMS v12 operational from Oct 2012
EVENT_VINTAGE = {
    "houston": {
        "harvey2017": {"year": 2017, "temporal_coverage": True},
        "imelda2019": {"year": 2019, "temporal_coverage": True},
        "beryl2024": {"year": 2024, "temporal_coverage": True},
    },
    "nyc": {
        "sandy2012": {"year": 2012, "temporal_coverage": False,
                      "reason": "Predates MRMS v12 (Oct 2012)"},
        "ida2021": {"year": 2021, "temporal_coverage": True},
        "henri2021": {"year": 2021, "temporal_coverage": True},
        "nyc2023": {"year": 2023, "temporal_coverage": True},
    },
    "southwest_florida": {
        "ian2022": {"year": 2022, "temporal_coverage": True},
        "helene2024": {"year": 2024, "temporal_coverage": True},
        "milton2024": {"year": 2024, "temporal_coverage": True},
    },
    "riverside_coachella": {
        "hilary2023": {"year": 2023, "temporal_coverage": True},
    },
    "new_orleans": {
        "katrina2005": {"year": 2005, "temporal_coverage": False,
                        "reason": "Predates MRMS v12 (Oct 2012)"},
        "isaac2012": {"year": 2012, "temporal_coverage": False,
                      "reason": "Predates MRMS v12 operational date"},
        "barry2019": {"year": 2019, "temporal_coverage": True},
        "ida2021": {"year": 2021, "temporal_coverage": True},
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_s3_client = None


def _get_s3() -> object:
    global _s3_client
    if _s3_client is None:
        _s3_client = get_s3_client()
    return _s3_client


def load_json(name: str, s3_prefix: str = RESULTS_PREFIX) -> dict | None:
    """Load a JSON from S3 (primary) or local mmar-input (fallback).

    Args:
        name: filename like 'diagnostics_r0.json' or 'lisa_results.json'
        s3_prefix: S3 key prefix (default: results/s035)
    """
    # Try S3 first
    s3 = _get_s3()
    key = f"{s3_prefix}/{name}"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        log.info("Loaded from s3://%s/%s", BUCKET, key)
        return data
    except Exception:
        pass

    # Fallback to local
    path = LOCAL_MMAR_DIR / name
    if path.exists():
        with open(path) as f:
            log.info("Loaded from local: %s", path)
            return json.load(f)

    log.warning("Could not load %s from S3 or local", name)
    return None


def load_parquet_s3(s3, key: str) -> pd.DataFrame | None:
    """Load parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load s3://%s/%s: %s", BUCKET, key, exc)
        return None


# ---------------------------------------------------------------------------
# Verdict 2: Ranking (Kendall tau-b)
# ---------------------------------------------------------------------------

def compute_ranking_verdict(s3) -> dict:
    """Ranking verdict: tau-b between R0/R1 orderings + fidelity-delta vs FAST.

    SUPPORTED if tau-b < 0.90 (meaningful reorder) AND fidelity-delta > 0.
    PROVISIONAL if reorder without fidelity improvement.
    NOT_LOAD_BEARING if tau-b >= 0.90.
    """
    log.info("=== RANKING VERDICT ===")
    per_scenario = []

    for scenario in SCENARIOS:
        r0_key = f"{RESULTS_PREFIX}/{level_prefix('r0')}_{scenario}_predictions.parquet"
        r1_key = f"{RESULTS_PREFIX}/{level_prefix('r1')}_{scenario}_predictions.parquet"

        r0_df = load_parquet_s3(s3, r0_key)
        r1_df = load_parquet_s3(s3, r1_key)

        if r0_df is None or r1_df is None:
            log.warning("Missing predictions for %s, skipping", scenario)
            per_scenario.append({
                "scenario": scenario,
                "error": "missing prediction parquets",
            })
            continue

        # Filter to primary target + solver, spatial_blocked split
        for df_name, df in [("r0", r0_df), ("r1", r1_df)]:
            log.info("  %s_%s predictions: %d rows, columns: %s",
                     df_name, scenario, len(df), list(df.columns)[:10])

        # Get unique events in this scenario
        events = sorted(r0_df.loc[
            (r0_df["target"] == PRIMARY_TARGET) &
            (r0_df["solver"] == PRIMARY_SOLVER),
            "event"
        ].unique()) if "event" in r0_df.columns else []

        if not events:
            # Fall back to ZCTA-level aggregation (no event column)
            result = _ranking_zcta_level(r0_df, r1_df, scenario)
            per_scenario.append(result)
            continue

        event_results = []
        for event in events:
            result = _ranking_event_level(r0_df, r1_df, scenario, event)
            if result:
                event_results.append(result)

        if not event_results:
            per_scenario.append({
                "scenario": scenario,
                "error": "no valid event-level comparisons",
            })
            continue

        # Aggregate across events for this scenario
        tau_bs = [r["tau_b"] for r in event_results if r.get("tau_b") is not None]
        mean_tau_b = float(np.mean(tau_bs)) if tau_bs else None

        per_scenario.append({
            "scenario": scenario,
            "n_events": len(event_results),
            "mean_tau_b": mean_tau_b,
            "events": event_results,
        })

    # Try to load FAST validation for fidelity-delta
    fast_data = load_json("fast_validation.json")

    fidelity_deltas = _compute_fidelity_deltas(fast_data) if fast_data else {}

    # Compute overall verdict
    all_tau_bs = []
    for sc in per_scenario:
        if sc.get("mean_tau_b") is not None:
            all_tau_bs.append(sc["mean_tau_b"])

    pooled_tau_b = float(np.mean(all_tau_bs)) if all_tau_bs else None

    if pooled_tau_b is None:
        verdict = "INSUFFICIENT"
        rationale = "No valid tau-b comparisons available"
    elif pooled_tau_b >= TAU_B_REORDER_THRESHOLD:
        verdict = "NOT_LOAD_BEARING"
        rationale = (f"tau-b={pooled_tau_b:.3f} >= {TAU_B_REORDER_THRESHOLD}: "
                     "context does not meaningfully reorder predictions")
    else:
        # Check fidelity-delta
        if fidelity_deltas:
            mean_fid = float(np.mean(list(fidelity_deltas.values())))
            if mean_fid > 0:
                verdict = "SUPPORTED"
                rationale = (f"tau-b={pooled_tau_b:.3f} < {TAU_B_REORDER_THRESHOLD} "
                             f"AND fidelity-delta={mean_fid:+.3f} > 0")
            else:
                verdict = "PROVISIONAL"
                rationale = (f"tau-b={pooled_tau_b:.3f} (meaningful reorder) "
                             f"but fidelity-delta={mean_fid:+.3f} <= 0")
        else:
            verdict = "PROVISIONAL"
            rationale = (f"tau-b={pooled_tau_b:.3f} (meaningful reorder) "
                         "but FAST data not yet available for fidelity-delta")

    return {
        "geometry": "ranking",
        "verdict": verdict,
        "rationale": rationale,
        "pooled_tau_b": pooled_tau_b,
        "threshold": TAU_B_REORDER_THRESHOLD,
        "fidelity_deltas": fidelity_deltas if fidelity_deltas else None,
        "per_scenario": per_scenario,
    }


def _ranking_event_level(
    r0_df: pd.DataFrame, r1_df: pd.DataFrame, scenario: str, event: str,
) -> dict | None:
    """Compute Kendall tau-b between R0 and R1 predictions for one event."""
    mask_r0 = (
        (r0_df["target"] == PRIMARY_TARGET) &
        (r0_df["solver"] == PRIMARY_SOLVER) &
        (r0_df["event"] == event)
    )
    mask_r1 = (
        (r1_df["target"] == PRIMARY_TARGET) &
        (r1_df["solver"] == PRIMARY_SOLVER) &
        (r1_df["event"] == event)
    )

    r0_evt = r0_df.loc[mask_r0].groupby("zcta_id", as_index=False)["y_pred"].mean()
    r1_evt = r1_df.loc[mask_r1].groupby("zcta_id", as_index=False)["y_pred"].mean()

    merged = r0_evt.merge(r1_evt, on="zcta_id", suffixes=("_r0", "_r1"))
    if len(merged) < 5:
        return None

    tau_b, p_val = stats.kendalltau(merged["y_pred_r0"], merged["y_pred_r1"])
    return {
        "event": event,
        "tau_b": float(tau_b),
        "p_value": float(p_val),
        "n_zctas": len(merged),
    }


def _ranking_zcta_level(
    r0_df: pd.DataFrame, r1_df: pd.DataFrame, scenario: str,
) -> dict:
    """Fallback: tau-b at ZCTA level without event stratification."""
    def _agg(df):
        mask = (df["target"] == PRIMARY_TARGET) & (df["solver"] == PRIMARY_SOLVER)
        return df.loc[mask].groupby("zcta_id", as_index=False)["y_pred"].mean()

    r0_agg = _agg(r0_df)
    r1_agg = _agg(r1_df)
    merged = r0_agg.merge(r1_agg, on="zcta_id", suffixes=("_r0", "_r1"))

    if len(merged) < 5:
        return {"scenario": scenario, "error": "too few ZCTAs for tau-b"}

    tau_b, p_val = stats.kendalltau(merged["y_pred_r0"], merged["y_pred_r1"])
    return {
        "scenario": scenario,
        "method": "zcta_collapsed",
        "tau_b": float(tau_b),
        "p_value": float(p_val),
        "n_zctas": len(merged),
        "mean_tau_b": float(tau_b),
    }



def _compute_fidelity_deltas(fast_data: dict) -> dict:
    """Extract fidelity-delta from FAST validation results.

    fidelity-delta = rho(R1, FAST) - rho(R0, FAST)
    Uses Spearman rho as proxy for Kendall tau-b from FAST validation.
    """
    deltas = {}
    for row in fast_data.get("validation_table", []):
        rho_r0 = row.get("rho_r0_fast")
        rho_r1 = row.get("rho_r1_fast")
        if rho_r0 is not None and rho_r1 is not None:
            key = f"{row['scenario']}/{row['event']}"
            deltas[key] = rho_r1 - rho_r0
    return deltas


# ---------------------------------------------------------------------------
# Verdict 3: Clustering (Moran's I)
# ---------------------------------------------------------------------------

def compute_clustering_verdict() -> dict:
    """Clustering verdict: residual Moran's I after spatial ladder level.

    SUPPORTED if Moran's I is indistinguishable from zero after R1.
    INSUFFICIENT if errors remain spatially clustered.
    """
    log.info("=== CLUSTERING VERDICT ===")
    lisa = load_json("lisa_results.json", s3_prefix=SIDECAR_PREFIX)
    if lisa is None:
        return {
            "geometry": "clustering",
            "verdict": "INSUFFICIENT",
            "rationale": "LISA results not available",
        }

    rollups = lisa.get("rollups", [])
    attenuations = lisa.get("attenuations", [])

    # Per-cell analysis: compare R0 vs R1 Moran's I
    per_cell = []
    for att in attenuations:
        scenario = att["scenario"]
        target = att["target"]

        # Get R1 Moran's I for this cell
        r1_moran = None
        r0_moran = None
        for r in rollups:
            if r["scenario"] == scenario and r["target"] == target:
                if r["level"] == "r1":
                    r1_moran = r["global_moran_I"]
                elif r["level"] == "r0":
                    r0_moran = r["global_moran_I"]

        # Moran's I near zero = no spatial autocorrelation (good)
        # Moran's I significantly positive = clustered residuals (bad)
        cell_verdict = "SUPPORTED" if (
            r1_moran is not None and abs(r1_moran) < 0.10
        ) else "INSUFFICIENT"

        delta_sign = att.get("delta_R0_R1_sign", "unknown")

        per_cell.append({
            "scenario": scenario,
            "target": target,
            "moran_I_r0": r0_moran,
            "moran_I_r1": r1_moran,
            "delta_R0_R1": att.get("delta_R0_R1"),
            "attenuation": delta_sign,
            "cell_verdict": cell_verdict,
        })

    n_supported = sum(1 for c in per_cell if c["cell_verdict"] == "SUPPORTED")
    n_total = len(per_cell)

    # Overall verdict: SUPPORTED only if majority of cells pass
    if n_total == 0:
        verdict = "INSUFFICIENT"
        rationale = "No cells available for clustering analysis"
    elif n_supported > n_total / 2:
        verdict = "SUPPORTED"
        rationale = (f"{n_supported}/{n_total} cells have Moran's I "
                     "indistinguishable from zero after R1")
    else:
        verdict = "INSUFFICIENT"
        rationale = (f"Only {n_supported}/{n_total} cells resolved spatial clustering; "
                     "residual spatial autocorrelation persists")

    # Attenuation summary
    n_attenuated = sum(1 for c in per_cell if c["attenuation"] == "attenuated")

    return {
        "geometry": "clustering",
        "verdict": verdict,
        "rationale": rationale,
        "n_cells": n_total,
        "n_supported": n_supported,
        "n_attenuated_R0_R1": n_attenuated,
        "attenuation_summary": lisa.get("attenuation_summary"),
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Verdict 4: Transfer (LEO retention)
# ---------------------------------------------------------------------------

def compute_transfer_verdict() -> dict:
    """Transfer verdict: LEO skill retention ratio.

    SUPPORTED if leave-event-out skill retains >= threshold of
    spatial-blocked skill. PROVISIONAL otherwise.
    """
    log.info("=== TRANSFER VERDICT ===")

    per_cell = []
    for level in ["r0", "r1", "r2"]:
        diag = load_json(f"diagnostics_{level}.json")
        if diag is None:
            continue
        for cell in diag.get("cells", []):
            dt = cell.get("diag_transfer")
            per_cell.append({
                "scenario": cell["scenario"],
                "target": cell["target"],
                "level": level,
                "diag_transfer": dt,
                "primary_metric": cell.get("primary_metric_value"),
                "has_transfer": dt is not None and dt > 0,
            })

    if not per_cell:
        return {
            "geometry": "transfer",
            "verdict": "INSUFFICIENT",
            "rationale": "No diagnostic data available",
        }

    # Focus on regression cells (primary target) at best level
    regression_cells = [c for c in per_cell
                        if c["target"] == PRIMARY_TARGET]

    # Per-scenario best-level transfer
    per_scenario = {}
    for c in regression_cells:
        sc = c["scenario"]
        if sc not in per_scenario:
            per_scenario[sc] = []
        per_scenario[sc].append(c)

    scenario_verdicts = []
    for sc, cells in per_scenario.items():
        # Take the latest level with data
        cells_sorted = sorted(cells, key=lambda x: x["level"], reverse=True)
        best = cells_sorted[0]
        dt = best["diag_transfer"]

        if dt is None:
            sv = "INSUFFICIENT"
            reason = "LEO split not available"
        elif dt >= TRANSFER_RETENTION_THRESHOLD:
            sv = "SUPPORTED"
            reason = f"diag_transfer={dt:.3f} >= {TRANSFER_RETENTION_THRESHOLD}"
        elif dt > 0:
            sv = "PROVISIONAL"
            reason = (f"diag_transfer={dt:.3f} > 0 but "
                      f"< {TRANSFER_RETENTION_THRESHOLD}")
        else:
            sv = "PROVISIONAL"
            reason = f"diag_transfer={dt:.3f} (LEO skill <= 0, clamped)"

        scenario_verdicts.append({
            "scenario": sc,
            "level": best["level"],
            "diag_transfer": dt,
            "verdict": sv,
            "rationale": reason,
        })

    n_supported = sum(1 for v in scenario_verdicts if v["verdict"] == "SUPPORTED")
    n_total = len(scenario_verdicts)

    if n_supported > n_total / 2:
        verdict = "SUPPORTED"
        rationale = (f"{n_supported}/{n_total} scenarios retain LEO skill "
                     f">= {TRANSFER_RETENTION_THRESHOLD}")
    elif any(v["verdict"] == "SUPPORTED" for v in scenario_verdicts):
        verdict = "PROVISIONAL"
        rationale = (f"Only {n_supported}/{n_total} scenarios meet retention "
                     "threshold; cross-event transfer is partial")
    else:
        verdict = "PROVISIONAL"
        rationale = ("No scenario retains sufficient LEO skill; "
                     "cross-event generalization is not demonstrated")

    return {
        "geometry": "transfer",
        "verdict": verdict,
        "rationale": rationale,
        "threshold": TRANSFER_RETENTION_THRESHOLD,
        "n_scenarios": n_total,
        "n_supported": n_supported,
        "per_scenario": scenario_verdicts,
        "all_cells": per_cell,
    }


# ---------------------------------------------------------------------------
# Verdict 5: Relational (W-matrix ablation)
# ---------------------------------------------------------------------------

# Pre-committed threshold: W-matrix RMSE degradation (Cohen's d) for
# the ablation to be considered load-bearing.
WMATRIX_EFFECT_THRESHOLD = 0.20  # |d| >= 0.20 = small-but-real effect


def _extract_fold_metrics(
    result_json: dict,
    target: str = PRIMARY_TARGET,
    solver: str = PRIMARY_SOLVER,
    split: str = "leave_event_out",
    metric: str = "rmse",
) -> list[float]:
    """Extract per-fold metric values from an R1 result JSON.

    Returns list of floats (one per fold) for the specified
    target/solver/split/metric combination.
    """
    vals = []
    for run in result_json.get("runs", []):
        if (run.get("target") == target
                and run.get("solver") == solver
                and run.get("split") == split):
            v = run.get("metrics", {}).get(metric)
            if v is not None and not math.isnan(v):
                vals.append(float(v))
    return vals


def compute_relational_verdict() -> dict:
    """Relational verdict: W-matrix ablation battery.

    Loads R1-full, R1-no-wlag, and R1-wlag-only results per scenario.
    Compares RMSE distributions (per-fold) via paired Wilcoxon + Cohen's d.
    W-matrix is LOAD_BEARING if removing it degrades RMSE significantly.

    Verdict vocabulary:
      SUPPORTED        -- majority of scenarios show load-bearing W-matrix
      PROVISIONAL      -- some scenarios show effect, not majority
      NOT_LOAD_BEARING -- no scenario shows meaningful W-matrix contribution
      PENDING          -- ablation results not yet available
      INSUFFICIENT     -- not enough data to compute
    """
    log.info("=== RELATIONAL VERDICT ===")

    scenario_verdicts = []
    missing_scenarios = []

    for scenario in SCENARIOS:
        full = load_json(f"r1_hydrology_{scenario}.json")
        no_wlag = load_json(f"r1_no_wlag_{scenario}.json")
        wlag_only = load_json(f"r1_wlag_only_{scenario}.json")

        if full is None:
            missing_scenarios.append((scenario, "r1_hydrology (full)"))
            continue
        if no_wlag is None and wlag_only is None:
            missing_scenarios.append((scenario, "r1_no_wlag + r1_wlag_only"))
            continue

        full_rmse = _extract_fold_metrics(full)
        no_wlag_rmse = _extract_fold_metrics(no_wlag) if no_wlag else []
        wlag_only_rmse = _extract_fold_metrics(wlag_only) if wlag_only else []

        sv = {
            "scenario": scenario,
            "full_mean_rmse": float(np.mean(full_rmse)) if full_rmse else None,
            "full_n_folds": len(full_rmse),
        }

        # Primary comparison: full vs no-wlag (removing W-matrix)
        if full_rmse and no_wlag_rmse and len(full_rmse) == len(no_wlag_rmse):
            full_arr = np.array(full_rmse)
            no_wlag_arr = np.array(no_wlag_rmse)
            delta = no_wlag_arr - full_arr  # positive = W-matrix helped

            pooled_std = np.sqrt(
                (np.var(full_arr, ddof=1) + np.var(no_wlag_arr, ddof=1)) / 2
            )
            cohens_d = float(np.mean(delta) / pooled_std) if pooled_std > 0 else 0.0

            # Paired Wilcoxon on RMSE (two-sided)
            try:
                stat, p_val = stats.wilcoxon(no_wlag_rmse, full_rmse,
                                             alternative="two-sided")
                p_val = float(p_val)
            except ValueError:
                # All differences zero or n < 6
                p_val = 1.0

            sv["no_wlag_mean_rmse"] = float(np.mean(no_wlag_arr))
            sv["no_wlag_n_folds"] = len(no_wlag_rmse)
            sv["rmse_delta_mean"] = float(np.mean(delta))
            sv["cohens_d"] = cohens_d
            sv["wilcoxon_p"] = p_val

            if abs(cohens_d) >= WMATRIX_EFFECT_THRESHOLD and p_val < 0.10:
                sv["verdict"] = "SUPPORTED"
                sv["rationale"] = (
                    f"Removing W-matrix degrades RMSE by d={cohens_d:.3f} "
                    f"(p={p_val:.4f}); W-matrix is load-bearing"
                )
            elif abs(cohens_d) >= WMATRIX_EFFECT_THRESHOLD:
                sv["verdict"] = "PROVISIONAL"
                sv["rationale"] = (
                    f"Effect size d={cohens_d:.3f} meets threshold but "
                    f"p={p_val:.4f} > 0.10; suggestive but not significant"
                )
            else:
                sv["verdict"] = "NOT_LOAD_BEARING"
                sv["rationale"] = (
                    f"d={cohens_d:.3f} < {WMATRIX_EFFECT_THRESHOLD}; "
                    "W-matrix does not meaningfully affect RMSE"
                )
        elif full_rmse and no_wlag_rmse:
            sv["verdict"] = "INSUFFICIENT"
            sv["rationale"] = (
                f"Fold count mismatch: full={len(full_rmse)}, "
                f"no_wlag={len(no_wlag_rmse)}"
            )
        else:
            sv["verdict"] = "PENDING"
            sv["rationale"] = "no-wlag ablation not yet available"

        # Secondary: wlag-only contribution (informational, not gating)
        if full_rmse and wlag_only_rmse:
            sv["wlag_only_mean_rmse"] = float(np.mean(wlag_only_rmse))
            sv["wlag_only_n_folds"] = len(wlag_only_rmse)

        scenario_verdicts.append(sv)

    # Handle all-missing case
    if not scenario_verdicts:
        return {
            "geometry": "relational",
            "verdict": "PENDING",
            "rationale": ("No ablation results available for any scenario. "
                          "Required: r1_no_wlag_{scenario}.json"),
            "missing": missing_scenarios,
        }

    # Aggregate across scenarios
    resolved = [v for v in scenario_verdicts if v["verdict"] not in
                ("PENDING", "INSUFFICIENT")]
    n_supported = sum(1 for v in resolved if v["verdict"] == "SUPPORTED")
    n_not_lb = sum(1 for v in resolved if v["verdict"] == "NOT_LOAD_BEARING")
    n_resolved = len(resolved)
    n_total = len(scenario_verdicts)
    n_pending = sum(1 for v in scenario_verdicts
                    if v["verdict"] in ("PENDING", "INSUFFICIENT"))

    if n_resolved == 0:
        verdict = "PENDING"
        rationale = (f"0/{n_total} scenarios have ablation results; "
                     "verdict deferred until ablation runs complete")
    elif n_supported > n_resolved / 2:
        verdict = "SUPPORTED"
        rationale = (f"{n_supported}/{n_resolved} scenarios show load-bearing "
                     f"W-matrix (|d| >= {WMATRIX_EFFECT_THRESHOLD})")
    elif n_supported > 0:
        verdict = "PROVISIONAL"
        rationale = (f"{n_supported}/{n_resolved} scenarios show effect; "
                     "W-matrix contribution is partial across metros")
    elif n_not_lb == n_resolved:
        verdict = "NOT_LOAD_BEARING"
        rationale = (f"0/{n_resolved} scenarios show meaningful W-matrix "
                     "contribution; relational geometry not supported")
    else:
        verdict = "PROVISIONAL"
        rationale = f"Mixed results across {n_resolved} resolved scenarios"

    if n_pending > 0:
        rationale += f" ({n_pending}/{n_total} scenarios still pending)"

    return {
        "geometry": "relational",
        "verdict": verdict,
        "rationale": rationale,
        "threshold_d": WMATRIX_EFFECT_THRESHOLD,
        "n_scenarios": n_total,
        "n_resolved": n_resolved,
        "n_supported": n_supported,
        "n_not_load_bearing": n_not_lb,
        "n_pending": n_pending,
        "per_scenario": scenario_verdicts,
        "missing_scenarios": missing_scenarios,
    }


# ---------------------------------------------------------------------------
# Verdict 6: Allocation (temporal coverage by vintage)
# ---------------------------------------------------------------------------

def compute_allocation_verdict() -> dict:
    """Allocation verdict: R2 temporal feature coverage by event vintage.

    SUPPORTED for events with temporal coverage (post-Oct 2012).
    INSUFFICIENT (evidence absent by vintage) for pre-MRMS events.
    The verdict is necessarily partial across the event panel.
    """
    log.info("=== ALLOCATION VERDICT ===")

    per_scenario = []
    total_events = 0
    covered_events = 0
    uncovered_events = 0

    for scenario, events in EVENT_VINTAGE.items():
        scenario_covered = 0
        scenario_uncovered = 0
        event_details = []

        for event, info in events.items():
            total_events += 1
            if info["temporal_coverage"]:
                covered_events += 1
                scenario_covered += 1
            else:
                uncovered_events += 1
                scenario_uncovered += 1

            event_details.append({
                "event": event,
                "year": info["year"],
                "temporal_coverage": info["temporal_coverage"],
                "reason": info.get("reason"),
            })

        per_scenario.append({
            "scenario": scenario,
            "n_events": len(events),
            "n_covered": scenario_covered,
            "n_uncovered": scenario_uncovered,
            "events": event_details,
        })

    # Check R2 certificates exist
    r2_certs = load_json("certificates_r2.json")
    has_r2 = r2_certs is not None and len(r2_certs.get("certificates", [])) > 0

    verdict = "PARTIAL"
    rationale = (
        f"{covered_events}/{total_events} events have temporal coverage "
        f"(post-MRMS v12 Oct 2012). {uncovered_events} events predate "
        "rainfall-timing sources. The verdict is necessarily partial -- "
        "a demonstration of the benchmark surfacing where a decision "
        "lacks support."
    )

    return {
        "geometry": "allocation",
        "verdict": verdict,
        "rationale": rationale,
        "total_events": total_events,
        "covered_events": covered_events,
        "uncovered_events": uncovered_events,
        "vintage_boundary": "MRMS v12 operational Oct 2012",
        "r2_certificates_available": has_r2,
        "per_scenario": per_scenario,
    }


# ---------------------------------------------------------------------------
# Prediction verdict (extract from money table)
# ---------------------------------------------------------------------------

def extract_prediction_verdict() -> dict:
    """Extract prediction verdict from existing money table.

    The prediction verdict (H2 Wilcoxon) is already computed in
    compute_uplift_table.py. We extract and reformat for consistency.
    """
    log.info("=== PREDICTION VERDICT (extract) ===")
    money = load_json("money_table.json")
    if money is None:
        return {
            "geometry": "prediction",
            "verdict": "INSUFFICIENT",
            "rationale": "Money table not available",
        }

    # Navigate to H2 evidence (money_table uses hypothesis_evidence.h2_evidence)
    h2 = (money.get("hypothesis_evidence", {}).get("h2_evidence", {})
           or money.get("hypothesis_tests", {}).get("h2_pooled_regression", {}))

    if not h2:
        return {
            "geometry": "prediction",
            "verdict": "INSUFFICIENT",
            "rationale": "H2 evidence not found in money table",
        }

    # H2 has a top-level verdict and a pooled sub-dict
    pooled = h2.get("pooled", h2)
    wilcoxon_p = pooled.get("wilcoxon_p_one_sided")
    cohens_d = pooled.get("cohens_d")
    verdict_raw = h2.get("verdict", pooled.get("verdict", "INCONCLUSIVE"))

    if verdict_raw == "PASS":
        verdict = "SUPPORTED"
    elif verdict_raw == "INCONCLUSIVE":
        verdict = "PROVISIONAL"
    else:
        verdict = "INSUFFICIENT"
    rationale = (f"Wilcoxon p={wilcoxon_p}, Cohen's d={cohens_d}; "
                 f"H2 verdict={verdict_raw}")

    return {
        "geometry": "prediction",
        "verdict": verdict,
        "rationale": rationale,
        "wilcoxon_p": wilcoxon_p,
        "cohens_d": cohens_d,
        "raw_verdict": verdict_raw,
        "source": "money_table.json / h2_pooled_regression",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Six-geometry verdicts")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would compute verdicts for 6 geometries")
        log.info("  Reads: mmar-input/*.json + prediction parquets from S3")
        log.info("  Writes: %s/verdicts.json", RESULTS_PREFIX)
        return 0

    global _s3_client
    _s3_client = get_s3_client()
    s3 = _s3_client

    prediction = extract_prediction_verdict()
    ranking = compute_ranking_verdict(s3)
    clustering = compute_clustering_verdict()
    transfer = compute_transfer_verdict()
    relational = compute_relational_verdict()
    allocation = compute_allocation_verdict()

    verdicts = [prediction, ranking, clustering, transfer, relational, allocation]

    result = {
        "phase": "8_verdicts",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_geometries": len(verdicts),
        "summary": {v["geometry"]: v["verdict"] for v in verdicts},
        "verdicts": verdicts,
    }

    # Write local copy (works on SageMaker and local)
    out_dir = Path("/tmp/verdicts")
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / "verdicts.json"
    with open(local_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Written to %s", local_file)

    if args.upload:
        key = f"{RESULTS_PREFIX}/verdicts.json"
        upload_json_result(s3, BUCKET, key, result)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Print summary
    print(f"\n{'='*60}")
    print("  SIX-GEOMETRY VERDICT TABLE")
    print(f"{'='*60}\n")
    for v in verdicts:
        status = v["verdict"]
        marker = {"SUPPORTED": "+", "PROVISIONAL": "~", "PARTIAL": "~",
                  "INSUFFICIENT": "-", "PENDING": "?",
                  "NOT_LOAD_BEARING": "o"}.get(status, " ")
        print(f"  [{marker}] {v['geometry']:20s}  {status:20s}")
        print(f"      {v['rationale']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
