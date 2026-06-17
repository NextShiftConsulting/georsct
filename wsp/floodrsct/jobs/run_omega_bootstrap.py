#!/usr/bin/env python3
"""run_omega_bootstrap.py -- DOE-C2a: Omega bootstrap per construct.

Resamples fold assignments B times per (scenario, construct) cell,
recomputes certificates each time, and derives omega (distributional
reliability) from bootstrap variance. Then computes P16 blended
quality (alpha_omega) and re-derives the divergence matrix.

Adapter script: same hex arch as run_five_construct_divergence.py.
Domain math in georsct.domain, orchestration in georsct.application.
This file handles S3 I/O, bootstrap resampling, and CLI.

Usage:
    python run_omega_bootstrap.py --scenario houston --upload
    python run_omega_bootstrap.py --scenario houston --n-bootstrap 50 --upload
    python run_omega_bootstrap.py --scenario houston --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Domain + ports (installed via rsct wheel)
from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructLabel,
    ConstructCertificate,
    compute_kappa_spatial,
)
from georsct.domain.construct_divergence_matrix import (
    CONSTRUCT_ORDER,
    build_divergence_matrix,
    summarize_divergence,
)
from georsct.domain.kappa_reconstruct import compute_kappa_reconstruct
from georsct.ports.model_fitter import EmbedResult, FitPredictResult, ModelFitter
from georsct.ports.construct_data_source import ConstructData, ConstructDataSource

# S3 infrastructure
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
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

RESULTS_PREFIX = "results/s035/doe_c2a"
CACHE_PREFIX = "results/s035/doe_c2a/cache"

# Frozen hyperparameters (ADR-014) -- same as DOE-C1
HGBDT_PARAMS = dict(
    loss="squared_error",
    max_iter=300,
    max_depth=6,
    min_samples_leaf=10,
    learning_rate=0.05,
)

# P16 uninformative prior
PRIOR = 0.5


# =========================================================================
# Reuse DOE-C1 data loading (import from five_construct harness helpers)
# =========================================================================

# Import data loading functions from the DOE-C1 harness.
# These are in the same jobs/ directory, co-deployed to SageMaker.
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
    FAST_RP_MAP,
    EVENT_RP_MAP,
)


# =========================================================================
# Bootstrap resampling
# =========================================================================

def _resample_folds(
    fold_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap: resample which folds appear (with replacement).

    Preserves spatial blocking structure. Returns new fold assignment
    array with the same shape as fold_ids, but some folds may appear
    multiple times and some may be absent.

    The model trains on "all folds except the test fold" -- when a fold
    appears twice in the resampled set, the test fold still has a clean
    holdout. The effect is that different bootstrap samples have different
    effective training set sizes and different spatial coverage.
    """
    unique_folds = sorted(set(fold_ids))
    n_folds = len(unique_folds)

    # Resample fold indices with replacement
    resampled = rng.choice(unique_folds, size=n_folds, replace=True)

    # Map original fold IDs to resampled positions
    # Each original fold gets assigned to its resampled position
    new_folds = np.zeros_like(fold_ids)
    for new_idx, old_fold in enumerate(resampled):
        mask = fold_ids == old_fold
        new_folds[mask] = new_idx

    return new_folds


def _certify_one_bootstrap(
    construct: ConstructLabel,
    features: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    region_ids: np.ndarray,
    region_order: tuple[str, ...],
    coords2d: np.ndarray,
    W_geo: sparse.csr_matrix,
    model_fitter: ModelFitter,
) -> dict:
    """Run one certification and return the certificate triple."""
    from georsct.application.use_cases.certify_constructs import (
        certify_single_construct,
    )

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
        n_baseline_trials=10,   # Reduced for bootstrap speed
        n_mantel_perms=0,       # Skip Mantel in bootstrap (expensive)
    )

    return {
        "forward_score": float(cert.forward_score),
        "kappa_spatial": float(cert.kappa_spatial),
        "kappa_reconstruct": float(cert.kappa_reconstruct),
        "target_available": cert.target_available,
    }


def _compute_omega(values: np.ndarray, clip_range: float = 1.0) -> float:
    """Compute omega = 1 - (std / clip_range), clamped to [0, 1].

    Higher omega = more stable (lower variance relative to scale).
    """
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    omega = 1.0 - (std / clip_range)
    return float(np.clip(omega, 0.0, 1.0))


def _compute_alpha_omega(
    alpha: float,
    omega: float,
    prior: float = PRIOR,
) -> float:
    """P16 blended quality: alpha_omega = omega * alpha + (1-omega) * prior."""
    if not np.isfinite(alpha) or not np.isfinite(omega):
        return float("nan")
    return omega * alpha + (1.0 - omega) * prior


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DOE-C2a: Omega bootstrap per construct"
    )
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--n-bootstrap", type=int, default=50,
                   help="Number of bootstrap iterations (default: 50)")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    B = args.n_bootstrap

    if args.dry_run:
        log.info("DRY RUN: omega bootstrap for %s (B=%d)", args.scenario, B)
        log.info("Constructs: %s", [c.name for c in ConstructLabel])
        return 0

    s3 = get_s3_client()

    # ---------------------------------------------------------------
    # Load data (same as DOE-C1)
    # ---------------------------------------------------------------
    log.info("Loading data for scenario: %s", args.scenario)
    event_df = _load_event_features(s3, args.scenario)
    event_df = _merge_shared_layers(s3, event_df)
    folds_df = _load_folds(s3, args.scenario)
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

    features = (
        event_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    fold_ids = event_df["fold"].to_numpy()
    region_ids = event_df["zcta_id"].astype(str).to_numpy()
    region_order = tuple(sorted(set(region_ids)))

    if not coords_df.empty:
        coords_df = coords_df[coords_df["zcta_id"].isin(region_order)]
        coord_map = {
            str(r.zcta_id): (r.lat, r.lon)
            for _, r in coords_df.iterrows()
        }
        coords2d = np.array([coord_map.get(r, (0.0, 0.0)) for r in region_order])
    else:
        coords2d = np.zeros((len(region_order), 2))

    if adj_df is not None:
        W_geo = _build_adjacency_csr(adj_df, region_order)
    else:
        W_geo = sparse.eye(len(region_order), format="csr")

    data_source = S3ConstructDataSource(s3, event_df, args.scenario)

    # ---------------------------------------------------------------
    # Bootstrap loop (parallelized across bootstrap samples)
    # ---------------------------------------------------------------
    n_jobs = int(os.environ.get("OMEGA_N_JOBS", os.cpu_count() or 1))
    log.info("Bootstrap parallelism: n_jobs=%d (cpus=%s)", n_jobs, os.cpu_count())

    rng = np.random.default_rng(args.seed)
    construct_results = {}

    for construct in CONSTRUCT_ORDER:
        log.info("=" * 60)
        log.info("Bootstrap: %s (B=%d, n_jobs=%d)", construct.name, B, n_jobs)

        cd = data_source.load_construct_target(construct, args.scenario)
        if not cd.available:
            log.warning("Construct %s unavailable: %s", construct.name, cd.reason)
            construct_results[construct.name] = {
                "available": False,
                "reason": cd.reason,
            }
            continue

        target = cd.target_values

        # Pre-generate all resampled folds (deterministic, RNG-ordered)
        resampled_folds_all = [_resample_folds(fold_ids, rng) for _ in range(B)]

        def _run_one_bootstrap(b: int) -> dict:
            seed_b = args.seed + b + 1
            model_fitter = HistGBDTModelFitter(seed=seed_b)
            return _certify_one_bootstrap(
                construct=construct,
                features=features,
                target=target,
                fold_ids=resampled_folds_all[b],
                region_ids=region_ids,
                region_order=region_order,
                coords2d=coords2d,
                W_geo=W_geo,
                model_fitter=model_fitter,
            )

        bootstrap_certs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one_bootstrap)(b) for b in range(B)
        )
        log.info("  %s: %d/%d bootstrap samples complete", construct.name, B, B)

        # Compute omega from bootstrap samples
        fwd_samples = np.array([c["forward_score"] for c in bootstrap_certs])
        ks_samples = np.array([c["kappa_spatial"] for c in bootstrap_certs])
        kr_samples = np.array([c["kappa_reconstruct"] for c in bootstrap_certs])

        omega_forward = _compute_omega(fwd_samples)
        omega_spatial = _compute_omega(ks_samples)
        omega_composite = min(omega_forward, omega_spatial) if (
            np.isfinite(omega_forward) and np.isfinite(omega_spatial)
        ) else float("nan")

        # Point estimates from DOE-C1 (mean of bootstrap = ~point estimate)
        fwd_finite = fwd_samples[np.isfinite(fwd_samples)]
        alpha_point = float(np.mean(fwd_finite)) if len(fwd_finite) > 0 else float("nan")

        # P16 blended quality
        alpha_omega = _compute_alpha_omega(alpha_point, omega_composite)

        # CI computation
        def _ci95(arr):
            finite = arr[np.isfinite(arr)]
            if len(finite) < 3:
                return (float("nan"), float("nan"))
            lo = float(np.percentile(finite, 2.5))
            hi = float(np.percentile(finite, 97.5))
            return (lo, hi)

        construct_results[construct.name] = {
            "available": True,
            "n_bootstrap": B,
            "forward_score": {
                "mean": float(np.nanmean(fwd_samples)),
                "std": float(np.nanstd(fwd_samples, ddof=1)) if len(fwd_finite) > 2 else float("nan"),
                "ci_95": _ci95(fwd_samples),
                "omega": omega_forward,
            },
            "kappa_spatial": {
                "mean": float(np.nanmean(ks_samples)),
                "std": float(np.nanstd(ks_samples, ddof=1)) if np.isfinite(ks_samples).sum() > 2 else float("nan"),
                "ci_95": _ci95(ks_samples),
                "omega": omega_spatial,
            },
            "kappa_reconstruct": {
                "mean": float(np.nanmean(kr_samples)),
                "std": float(np.nanstd(kr_samples, ddof=1)) if np.isfinite(kr_samples).sum() > 2 else float("nan"),
                "ci_95": _ci95(kr_samples),
            },
            "omega_composite": omega_composite,
            "alpha_point": alpha_point,
            "alpha_omega": alpha_omega,
            "prior": PRIOR,
            "bootstrap_samples": [
                {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                 for k, v in c.items()}
                for c in bootstrap_certs
            ],
        }

        log.info(
            "  %s: omega_fwd=%.3f  omega_spat=%.3f  omega_comp=%.3f  "
            "alpha=%.3f  alpha_omega=%.3f",
            construct.name, omega_forward, omega_spatial, omega_composite,
            alpha_point, alpha_omega,
        )

    # ---------------------------------------------------------------
    # Build blended divergence matrix
    # ---------------------------------------------------------------
    log.info("Building blended divergence matrix")

    # Create certificates using alpha_omega instead of raw forward_score
    blended_certs = []
    for construct in CONSTRUCT_ORDER:
        cr = construct_results.get(construct.name, {})
        if not cr.get("available", False):
            blended_certs.append(ConstructCertificate.missing(construct, "unavailable"))
            continue

        blended_certs.append(ConstructCertificate.from_scores(
            construct=construct,
            target_column=CONSTRUCT_TARGET_COLUMNS.get(construct, ""),
            forward_score=float(np.clip(cr["alpha_omega"], 0.0, 1.0)),
            kappa_spatial=cr["kappa_spatial"]["mean"],
            kappa_reconstruct=cr["kappa_reconstruct"]["mean"],
            morans_i=float("nan"),  # Not tracked per bootstrap
            n_regions=len(region_order),
            n_observations=len(features),
            n_finite_targets=int(np.isfinite(features[:, 0]).sum()),
        ))

    blended_dm = build_divergence_matrix(blended_certs, geography_id=args.scenario)
    blended_summary = summarize_divergence(blended_dm)

    # ---------------------------------------------------------------
    # Assemble output
    # ---------------------------------------------------------------
    result = {
        "doe_id": "DOE-C2a",
        "phase": "omega_bootstrap",
        "scenario": args.scenario,
        "n_bootstrap": B,
        "seed": args.seed,
        "n_features": len(feature_cols),
        "model": "HistGradientBoostingRegressor",
        "model_params": HGBDT_PARAMS,
        "prior": PRIOR,
        "constructs": construct_results,
        "blended_divergence": blended_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Print summary
    print()
    print("=" * 72)
    print("DOE-C2a Omega Bootstrap: %s (B=%d)" % (args.scenario, B))
    print("=" * 72)
    for construct in CONSTRUCT_ORDER:
        cr = construct_results.get(construct.name, {})
        if not cr.get("available", False):
            print("  %-12s  MISSING" % construct.name)
            continue
        print(
            "  %-12s  omega=%.3f  alpha=%.3f  alpha_omega=%.3f  "
            "CI_fwd=[%.3f, %.3f]" % (
                construct.name,
                cr["omega_composite"],
                cr["alpha_point"],
                cr["alpha_omega"],
                cr["forward_score"]["ci_95"][0],
                cr["forward_score"]["ci_95"][1],
            )
        )
    print("=" * 72)

    # ---------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results" / "doe_c2a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap samples cache (full detail) — extract BEFORE stripping
    boot_rows = []
    for construct in CONSTRUCT_ORDER:
        cr = construct_results.get(construct.name, {})
        if not cr.get("available", False):
            continue
        for b, sample in enumerate(cr.get("bootstrap_samples", [])):
            boot_rows.append({
                "scenario": args.scenario,
                "construct": construct.value,
                "bootstrap_idx": b,
                "forward_score": sample["forward_score"],
                "kappa_spatial": sample["kappa_spatial"],
                "kappa_reconstruct": sample["kappa_reconstruct"],
            })

    # Strip bootstrap_samples from the JSON for size (keep in parquet)
    import copy
    result_slim = copy.deepcopy(result)
    for cname in result_slim.get("constructs", {}):
        cr = result_slim["constructs"][cname]
        if isinstance(cr, dict) and "bootstrap_samples" in cr:
            cr["bootstrap_samples"] = "see cache parquet"

    local_file = out_dir / ("omega_bootstrap_%s.json" % args.scenario)
    with open(local_file, "w") as f:
        json.dump(result_slim, f, indent=2, default=str)
    log.info("Written local: %s", local_file)
    boot_df = pd.DataFrame(boot_rows)

    # Omega summary table
    omega_rows = []
    for construct in CONSTRUCT_ORDER:
        cr = construct_results.get(construct.name, {})
        if not cr.get("available", False):
            continue
        omega_rows.append({
            "scenario": args.scenario,
            "construct": construct.value,
            "omega_forward": cr["forward_score"]["omega"],
            "omega_spatial": cr["kappa_spatial"]["omega"],
            "omega_composite": cr["omega_composite"],
            "alpha_point": cr["alpha_point"],
            "alpha_omega": cr["alpha_omega"],
            "forward_mean": cr["forward_score"]["mean"],
            "forward_std": cr["forward_score"]["std"],
            "forward_ci_lo": cr["forward_score"]["ci_95"][0],
            "forward_ci_hi": cr["forward_score"]["ci_95"][1],
            "kappa_spatial_mean": cr["kappa_spatial"]["mean"],
            "kappa_spatial_std": cr["kappa_spatial"]["std"],
        })
    omega_df = pd.DataFrame(omega_rows)

    # Save local cache
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not boot_df.empty:
        boot_df.to_parquet(cache_dir / ("bootstrap_samples_%s.parquet" % args.scenario), index=False)
    if not omega_df.empty:
        omega_df.to_parquet(cache_dir / ("omega_table_%s.parquet" % args.scenario), index=False)

    # Upload to S3
    if args.upload:
        json_key = "%s/omega_bootstrap_%s.json" % (RESULTS_PREFIX, args.scenario)
        upload_json_result(s3, BUCKET, json_key, result_slim)
        log.info("Uploaded s3://%s/%s", BUCKET, json_key)

        if not boot_df.empty:
            buf = io.BytesIO()
            boot_df.to_parquet(buf, index=False)
            buf.seek(0)
            key = "%s/bootstrap_samples_%s.parquet" % (CACHE_PREFIX, args.scenario)
            s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
            log.info("Uploaded s3://%s/%s", BUCKET, key)

        if not omega_df.empty:
            buf = io.BytesIO()
            omega_df.to_parquet(buf, index=False)
            buf.seek(0)
            key = "%s/omega_table_%s.parquet" % (CACHE_PREFIX, args.scenario)
            s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
            log.info("Uploaded s3://%s/%s", BUCKET, key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
