#!/usr/bin/env python3
"""run_temporal_prior.py -- DOE-C2b: Sequential certification with P16 hint.

Tests whether a prior event's certificate improves the next event's
certification via P16 blending (alpha_omega = omega * raw + (1-omega) * prior).

Two arms per (scenario, construct, event):
  A (independent): certify each event independently.
  B (sequential):  use prior event's certificate as P16 hint.

Usage:
    python run_temporal_prior.py --scenario houston --upload
    python run_temporal_prior.py --scenario nyc --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructLabel,
    compute_kappa_spatial,
)
from georsct.ports.model_fitter import ModelFitter

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result
from run_five_construct_divergence import (
    _load_event_features,
    _merge_shared_layers,
    _load_folds,
    _load_coords,
    _load_adjacency_df,
    _build_adjacency_csr,
    _select_features,
    HistGBDTModelFitter,
    S3ConstructDataSource,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

RESULTS_PREFIX = "results/s035/doe_c2b"
CACHE_PREFIX = "results/s035/doe_c2b/cache"

# Scenarios with multiple events (chronological order)
EVENT_ORDER = {
    "houston": ["harvey2017", "imelda2019", "beryl2024"],
    "nyc": ["sandy2012", "henri2021", "ida2021", "nyc_flood_2023"],
}

# Constructs tested in DOE-C2b (available across events in both scenarios)
TARGET_CONSTRUCTS = [
    ConstructLabel.NFIP,
    ConstructLabel.FEMA,
    ConstructLabel.JRC,
]

SCENARIOS = list(EVENT_ORDER.keys())


# =========================================================================
# Per-event certification
# =========================================================================

def _certify_event(
    construct: ConstructLabel,
    features: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    region_ids: np.ndarray,
    region_order: tuple[str, ...],
    coords2d: np.ndarray,
    W_geo: sparse.csr_matrix,
    seed: int = 42,
) -> dict:
    """Certify a single (construct, event) cell. Returns certificate dict."""
    from georsct.application.use_cases.certify_constructs import (
        certify_single_construct,
    )

    model_fitter = HistGBDTModelFitter(seed=seed)
    cert = certify_single_construct(
        construct=construct,
        features=features,
        target=target,
        fold_ids=fold_ids,
        region_ids=region_ids,
        region_order=region_order,
        coords2d=coords2d,
        W_geo=W_geo,
        model_fitter=model_fitter,
        n_baseline_trials=10,
        n_mantel_perms=0,
    )

    return {
        "forward_score": float(cert.forward_score),
        "kappa_spatial": float(cert.kappa_spatial),
        "kappa_reconstruct": float(cert.kappa_reconstruct),
        "target_available": cert.target_available,
    }


def _p16_blend(raw_alpha: float, prior_alpha: float, omega: float) -> float:
    """P16 blended quality: alpha_omega = omega * raw + (1 - omega) * prior."""
    if not np.isfinite(raw_alpha) or not np.isfinite(prior_alpha) or not np.isfinite(omega):
        return float("nan")
    return omega * raw_alpha + (1.0 - omega) * prior_alpha


# =========================================================================
# Main
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DOE-C2b: Temporal prior -- sequential certification with P16 hint"
    )
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scenario = args.scenario
    events = EVENT_ORDER[scenario]

    if args.dry_run:
        log.info("DRY RUN: temporal prior for %s (events: %s)", scenario, events)
        log.info("Constructs: %s", [c.name for c in TARGET_CONSTRUCTS])
        return 0

    s3 = get_s3_client()

    # ---------------------------------------------------------------
    # Load omega from DOE-C2a
    # ---------------------------------------------------------------
    log.info("Loading omega from DOE-C2a")
    omega_key = "results/s035/doe_c2a/omega_bootstrap_%s.json" % scenario
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=omega_key)
        omega_data = json.loads(resp["Body"].read())
    except Exception as e:
        log.error("Failed to load omega: %s", e)
        return 1

    omega_map = {}
    for cname, cdata in omega_data.get("constructs", {}).items():
        if isinstance(cdata, dict) and cdata.get("available"):
            omega_map[cname] = cdata.get("omega_composite", float("nan"))
    log.info("Omega map: %s", {k: "%.3f" % v if np.isfinite(v) else "NaN" for k, v in omega_map.items()})

    # ---------------------------------------------------------------
    # Load data (same pattern as DOE-C1/C2a)
    # ---------------------------------------------------------------
    log.info("Loading data for scenario: %s", scenario)
    event_df = _load_event_features(s3, scenario)
    event_df = _merge_shared_layers(s3, event_df)
    folds_df = _load_folds(s3, scenario)
    coords_df = _load_coords(s3)
    adj_df = _load_adjacency_df(s3)

    if not folds_df.empty and "fold" not in event_df.columns:
        folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
        fold_map = folds_df[["zcta_id", "fold"]].drop_duplicates(subset="zcta_id")
        event_df = event_df.merge(fold_map, on="zcta_id", how="left")

    if "fold" not in event_df.columns:
        log.info("No folds found, creating hash-based folds")
        event_df["fold"] = event_df["zcta_id"].apply(lambda z: hash(z) % 5)

    feature_cols = _select_features(event_df, "obs_nfip_event_claims")
    log.info("Selected %d features", len(feature_cols))

    data_source = S3ConstructDataSource(s3, event_df, scenario)

    # Build geo structures (shared across events -- same ZCTAs)
    all_region_ids = event_df["zcta_id"].astype(str).unique()
    all_region_order = tuple(sorted(all_region_ids))

    if not coords_df.empty:
        coords_df_sub = coords_df[coords_df["zcta_id"].isin(all_region_order)]
        coord_map = {
            str(r.zcta_id): (r.lat, r.lon)
            for _, r in coords_df_sub.iterrows()
        }
        coords2d = np.array([coord_map.get(r, (0.0, 0.0)) for r in all_region_order])
    else:
        coords2d = np.zeros((len(all_region_order), 2))

    if adj_df is not None:
        W_geo = _build_adjacency_csr(adj_df, all_region_order)
    else:
        W_geo = sparse.eye(len(all_region_order), format="csr")

    log.info("Adjacency: %d x %d, %d edges", W_geo.shape[0], W_geo.shape[1], W_geo.nnz)

    # ---------------------------------------------------------------
    # Certify each (construct, event) -- both arms
    # ---------------------------------------------------------------
    n_jobs = int(os.environ.get("TEMPORAL_N_JOBS", os.cpu_count() or 1))
    log.info("Parallelism: n_jobs=%d", n_jobs)

    results = {}

    for construct in TARGET_CONSTRUCTS:
        cname = construct.name
        omega = omega_map.get(cname, float("nan"))
        log.info("=" * 60)
        log.info("Construct: %s  omega=%.3f  events=%d",
                 cname, omega if np.isfinite(omega) else float("nan"), len(events))

        event_certs = {}
        prior_cert = None

        for event in events:
            log.info("  Event: %s", event)

            # Filter to this event
            edf = event_df[event_df["event"] == event].copy()
            if edf.empty:
                log.warning("    No rows for event %s, skipping", event)
                event_certs[event] = {"available": False, "reason": "no rows"}
                continue

            # Get target for this construct + event
            cd = data_source.load_construct_target(construct, scenario, event_id=event)
            if not cd.available:
                log.warning("    Construct %s unavailable for %s: %s",
                            cname, event, cd.reason)
                event_certs[event] = {"available": False, "reason": cd.reason}
                continue

            target = cd.target_values

            # Build per-event arrays
            e_features = (
                edf[feature_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            e_fold_ids = edf["fold"].to_numpy()
            e_region_ids = edf["zcta_id"].astype(str).to_numpy()
            e_region_order = tuple(sorted(set(e_region_ids)))

            # Subset coords and adjacency to this event's ZCTAs
            idx_map = {r: i for i, r in enumerate(all_region_order)}
            e_indices = [idx_map[r] for r in e_region_order if r in idx_map]
            e_coords2d = coords2d[e_indices] if e_indices else np.zeros((len(e_region_order), 2))

            if len(e_indices) > 0:
                e_W = W_geo[np.ix_(e_indices, e_indices)]
            else:
                e_W = sparse.eye(len(e_region_order), format="csr")

            # --- Arm A: Independent ---
            cert_independent = _certify_event(
                construct=construct,
                features=e_features,
                target=target,
                fold_ids=e_fold_ids,
                region_ids=e_region_ids,
                region_order=e_region_order,
                coords2d=e_coords2d,
                W_geo=e_W,
                seed=args.seed,
            )

            # --- Arm B: Sequential P16 ---
            raw_fwd = cert_independent["forward_score"]
            if prior_cert is not None and np.isfinite(omega):
                prior_fwd = prior_cert["forward_score"]
                blended_fwd = _p16_blend(raw_fwd, prior_fwd, omega)
            else:
                # First event or no omega -- blended = raw
                blended_fwd = raw_fwd

            cert_sequential = {
                **cert_independent,
                "blended_forward_score": blended_fwd,
                "prior_forward_score": prior_cert["forward_score"] if prior_cert else None,
                "omega_used": omega if np.isfinite(omega) else None,
                "is_first_event": prior_cert is None,
            }

            event_certs[event] = {
                "available": True,
                "n_rows": len(edf),
                "n_zctas": len(e_region_order),
                "arm_a_independent": cert_independent,
                "arm_b_sequential": cert_sequential,
            }

            log.info(
                "    Arm A (indep): fwd=%.3f  kappa_s=%.3f",
                cert_independent["forward_score"],
                cert_independent["kappa_spatial"],
            )
            log.info(
                "    Arm B (seq):   fwd_raw=%.3f  fwd_blended=%.3f  "
                "prior=%.3f  omega=%.3f",
                raw_fwd, blended_fwd,
                prior_cert["forward_score"] if prior_cert else float("nan"),
                omega if np.isfinite(omega) else float("nan"),
            )

            # Update prior for next event
            prior_cert = cert_independent

        # Compute summary statistics across events
        indep_fwds = [
            ec["arm_a_independent"]["forward_score"]
            for ec in event_certs.values()
            if isinstance(ec, dict) and ec.get("available")
        ]
        seq_fwds = [
            ec["arm_b_sequential"]["blended_forward_score"]
            for ec in event_certs.values()
            if isinstance(ec, dict) and ec.get("available")
        ]

        indep_arr = np.array(indep_fwds)
        seq_arr = np.array(seq_fwds)

        summary = {
            "n_events": len(events),
            "n_available": len(indep_fwds),
            "omega": omega if np.isfinite(omega) else None,
            "arm_a_mean_forward": float(np.mean(indep_arr)) if len(indep_arr) > 0 else None,
            "arm_a_std_forward": float(np.std(indep_arr, ddof=1)) if len(indep_arr) > 1 else None,
            "arm_b_mean_forward": float(np.mean(seq_arr)) if len(seq_arr) > 0 else None,
            "arm_b_std_forward": float(np.std(seq_arr, ddof=1)) if len(seq_arr) > 1 else None,
            "variance_reduction": None,
            "leakage_flag": False,
        }

        # AC-C2b-1: Does sequential blending reduce variance?
        if len(indep_arr) > 1 and len(seq_arr) > 1:
            std_a = float(np.std(indep_arr, ddof=1))
            std_b = float(np.std(seq_arr, ddof=1))
            summary["variance_reduction"] = (std_a - std_b) / std_a if std_a > 0 else 0.0

        # AC-C2b-2: Does blended score exceed actual holdout?
        for i, event in enumerate(events):
            ec = event_certs.get(event, {})
            if not isinstance(ec, dict) or not ec.get("available"):
                continue
            blended = ec["arm_b_sequential"]["blended_forward_score"]
            actual = ec["arm_a_independent"]["forward_score"]
            if np.isfinite(blended) and np.isfinite(actual) and blended > actual + 0.05:
                summary["leakage_flag"] = True
                log.warning(
                    "  LEAKAGE FLAG: %s %s blended (%.3f) > actual (%.3f)",
                    cname, event, blended, actual,
                )

        results[cname] = {
            "events": event_certs,
            "summary": summary,
        }

    # ---------------------------------------------------------------
    # Acceptance criteria
    # ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("Acceptance Criteria")
    log.info("=" * 60)

    ac_results = {}

    # AC-C2b-1: Sequential reduces variance for NFIP
    nfip_summary = results.get("NFIP", {}).get("summary", {})
    vr = nfip_summary.get("variance_reduction")
    ac_results["AC_C2b_1"] = {
        "description": "Sequential blending reduces certificate variance for NFIP",
        "pass": vr is not None and vr > 0,
        "variance_reduction": vr,
    }
    log.info("  AC-C2b-1 (NFIP variance reduction): %s (vr=%.3f)",
             "PASS" if ac_results["AC_C2b_1"]["pass"] else "FAIL",
             vr if vr is not None else float("nan"))

    # AC-C2b-2: No leakage
    any_leakage = any(
        r.get("summary", {}).get("leakage_flag", False)
        for r in results.values()
    )
    ac_results["AC_C2b_2"] = {
        "description": "No construct leakage through time",
        "pass": not any_leakage,
    }
    log.info("  AC-C2b-2 (no leakage): %s",
             "PASS" if not any_leakage else "FAIL")

    # AC-C2b-3: FEMA prior less informative than NFIP
    fema_vr = results.get("FEMA", {}).get("summary", {}).get("variance_reduction")
    nfip_vr = nfip_summary.get("variance_reduction")
    ac_results["AC_C2b_3"] = {
        "description": "FEMA prior less informative than NFIP prior",
        "pass": (fema_vr is not None and nfip_vr is not None and fema_vr < nfip_vr),
        "fema_vr": fema_vr,
        "nfip_vr": nfip_vr,
    }
    log.info("  AC-C2b-3 (FEMA < NFIP): %s (fema_vr=%.3f, nfip_vr=%.3f)",
             "PASS" if ac_results["AC_C2b_3"]["pass"] else "FAIL",
             fema_vr if fema_vr is not None else float("nan"),
             nfip_vr if nfip_vr is not None else float("nan"))

    # ---------------------------------------------------------------
    # Assemble output
    # ---------------------------------------------------------------
    result = {
        "doe_id": "DOE-C2b",
        "phase": "temporal_prior",
        "scenario": scenario,
        "events": events,
        "constructs_tested": [c.name for c in TARGET_CONSTRUCTS],
        "seed": args.seed,
        "results": results,
        "acceptance_criteria": ac_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Print summary table
    print()
    print("=" * 72)
    print("DOE-C2b Temporal Prior: %s (%d events)" % (scenario, len(events)))
    print("=" * 72)
    for construct in TARGET_CONSTRUCTS:
        cr = results.get(construct.name, {})
        s = cr.get("summary", {})
        print(
            "  %-6s  omega=%.3f  indep_std=%.3f  seq_std=%.3f  "
            "var_red=%.3f  leak=%s" % (
                construct.name,
                s.get("omega") or float("nan"),
                s.get("arm_a_std_forward") or float("nan"),
                s.get("arm_b_std_forward") or float("nan"),
                s.get("variance_reduction") or float("nan"),
                "YES" if s.get("leakage_flag") else "no",
            )
        )
    print("=" * 72)

    # ---------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------
    if args.upload:
        out_key = "%s/temporal_prior_%s.json" % (RESULTS_PREFIX, scenario)
        upload_json_result(s3, BUCKET, out_key, result)
        log.info("Uploaded: s3://%s/%s", BUCKET, out_key)

        # Cache: per-event certificate table
        cert_rows = []
        for cname, cr in results.items():
            for event, ec in cr.get("events", {}).items():
                if not isinstance(ec, dict) or not ec.get("available"):
                    continue
                arm_a = ec["arm_a_independent"]
                arm_b = ec["arm_b_sequential"]
                cert_rows.append({
                    "scenario": scenario,
                    "construct": cname,
                    "event": event,
                    "arm": "independent",
                    "forward_score": arm_a["forward_score"],
                    "kappa_spatial": arm_a["kappa_spatial"],
                    "kappa_reconstruct": arm_a["kappa_reconstruct"],
                })
                cert_rows.append({
                    "scenario": scenario,
                    "construct": cname,
                    "event": event,
                    "arm": "sequential_p16",
                    "forward_score": arm_b["blended_forward_score"],
                    "kappa_spatial": arm_a["kappa_spatial"],
                    "kappa_reconstruct": arm_a["kappa_reconstruct"],
                    "prior_forward_score": arm_b.get("prior_forward_score"),
                    "omega_used": arm_b.get("omega_used"),
                })

        if cert_rows:
            cache_key = "%s/sequential_certificates_%s.parquet" % (CACHE_PREFIX, scenario)
            buf = pd.DataFrame(cert_rows).to_parquet(index=False)
            # upload parquet bytes
            import io
            pq_buf = io.BytesIO()
            pd.DataFrame(cert_rows).to_parquet(pq_buf, index=False)
            pq_buf.seek(0)
            s3.put_object(Bucket=BUCKET, Key=cache_key, Body=pq_buf.getvalue())
            log.info("Uploaded cache: s3://%s/%s", BUCKET, cache_key)
    else:
        # Write locally
        out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results" / "doe_c2b"
        out_dir.mkdir(parents=True, exist_ok=True)
        local_file = out_dir / ("temporal_prior_%s.json" % scenario)
        with open(local_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Written local: %s", local_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
