#!/usr/bin/env python3
"""
run_s019a.py -- S019A Certificate Invariance Gradient

7 x 3 x 3 factorial: 7 embeddings x 3 solvers x 3 targets.
63 cells, each with 5-fold county-holdout CV = 315 fits.

Embeddings (7 arms):
  - pca_v1:          PCA32 on ACS features (curated baseline)
  - spatial_lag_v1:  ACS + neighbor spatial lag (spatial autocorrelation)
  - gnn_v2:          GraphSAGE latents (learned spatial representation)
  - acs_raw:         Full ACS features, scaled, no PCA (tests PCA information loss)
  - geo_spatial:     Geographic context only (flood, HIFLD, SVI, drive-time, lat/lon)
  - noisy_control:   PCA32 + N(0,1) noise (floor calibration -- gates MUST reject)
  - domain_features: All available features (ACS + lag + SVI + flood + HIFLD + drive)

Theory kappa (D*/D) via RegressionKappaEvaluator with leave-one-out
references (6 refs per embedding). Proxy kappa R*(1-N) computed in
parallel for comparison.

Paired gate evaluation: flat (CONUS27) + Oobleck (ADR-023 sigmoidal).
Full ADR-024 enforcement provenance on every gate result.

Data: geocert v23.0.2 (georsct_table.parquet, 106 columns, ~32k ZCTAs)

Compute profile:
  - Instance: ml.m5.2xlarge (8 vCPU, 32 GB RAM)
  - Image: pytorch-training:2.9.0-cpu-py312 (no GPU needed)
  - Memory: ~4 GB peak (32k rows x 106 cols, 7 embeddings, 8 joblib workers)
  - Parallelism: 15 groups/target (3 solvers x 5 folds), n_jobs=8
  - Timeout: 6 hours (21600s)
  - Checkpointing: per-target S3 checkpoint for crash recovery

Paper reference: GeoRSCT V3, Section 3.5 + Appendix F (S019A).
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Handle both SageMaker and local layouts
_here = Path(__file__).parent
for _base in [_here, _here.parent]:
    _p = str(_base)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# yrsn imports
from yrsn.core.certificates.estimate import CPGatekeeperInput
from yrsn_controlplane import SequentialGatekeeper, GatekeeperConfig, get_preset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S019A_TARGETS = ["diabetes", "population_density", "elevation"]
EMBEDDINGS = [
    "pca_v1", "spatial_lag_v1", "gnn_v2",
    "acs_raw", "geo_spatial", "noisy_control", "domain_features",
]
SOLVERS = ["histgbdt", "ridge", "mlp"]
N_FOLDS = 5
SEED = int(os.environ.get("S019A_SEED", "42"))


# ---------------------------------------------------------------------------
# Checkpoint helpers -- survive crashes via direct S3 writes
# ---------------------------------------------------------------------------

CHECKPOINT_BUCKET = os.environ.get("CHECKPOINT_BUCKET", "swarm-yrsn-datasets")
CHECKPOINT_PREFIX = os.environ.get(
    "CHECKPOINT_PREFIX",
    f"rsct_curriculum/series_019_v2/results/s019a/seed_{SEED}/checkpoints",
)


def _s3_checkpoint():
    """Lazy S3 client (IAM role in container)."""
    import boto3
    return boto3.client("s3", region_name="us-east-1")


def _save_checkpoint(target: str, data: list):
    try:
        key = f"{CHECKPOINT_PREFIX}/{target}.json"
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        _s3_checkpoint().put_object(Bucket=CHECKPOINT_BUCKET, Key=key, Body=body)
        log.info("  checkpoint -> s3://%s/%s", CHECKPOINT_BUCKET, key)
    except Exception as e:
        log.warning("  checkpoint FAILED for %s: %s", target, e)


def _load_completed_targets() -> set:
    try:
        prefix = f"{CHECKPOINT_PREFIX}/"
        resp = _s3_checkpoint().list_objects_v2(Bucket=CHECKPOINT_BUCKET, Prefix=prefix)
        completed = set()
        for obj in resp.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.endswith(".json"):
                completed.add(name[:-5])
        if completed:
            log.info("Resuming: found %d completed checkpoints", len(completed))
        return completed
    except Exception as e:
        log.warning("Could not check for existing checkpoints: %s", e)
        return set()


def _load_checkpoint(target: str) -> list:
    try:
        key = f"{CHECKPOINT_PREFIX}/{target}.json"
        resp = _s3_checkpoint().get_object(Bucket=CHECKPOINT_BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Two gatekeeper configs: flat + Oobleck (paired evaluation)
# ---------------------------------------------------------------------------

def _make_gatekeepers():
    """Create flat and oobleck gatekeepers for paired evaluation."""
    flat_config = get_preset("geospatial-conus27")

    oobleck_config = GatekeeperConfig(
        N_thr=flat_config.N_thr,
        alpha_min=flat_config.alpha_min,
        c_min=flat_config.c_min,
        gate_2_require_coherence=flat_config.gate_2_require_coherence,
        sigma_thr=flat_config.sigma_thr,
        kappa_base=flat_config.kappa_base,
        lambda_turbulence=0.4,       # ADR-023 delta_kappa
        steepness=10.0,              # ADR-023 sigmoidal steepness
        sigma_c=0.35,                # ADR-023 inflection point
        epsilon_L=flat_config.epsilon_L,
        enable_gate_3b=flat_config.enable_gate_3b,
        r_bar_min=flat_config.r_bar_min,
        gate_3b_action=flat_config.gate_3b_action,
        kappa_L_min=flat_config.kappa_L_min,
    )

    return {
        "flat": SequentialGatekeeper(flat_config),
        "oobleck": SequentialGatekeeper(oobleck_config),
    }


# ---------------------------------------------------------------------------
# Gate evaluation for a single result dict
# ---------------------------------------------------------------------------

def _evaluate_gates(result: dict, gatekeepers: dict) -> dict:
    """Evaluate one embedding result under both gatekeepers using theory kappa."""
    cert_input = CPGatekeeperInput(
        alpha=result["alpha"],
        kappa_compat=result["theory_kappa"],
        sigma=result["sigma"],
        source_mode="direct",
        evidence={
            "N": result["N"],
            "R": result["R"],
            "S": result["S_sup"],
            "noise_admissibility": result["N"],
            "omega": result["omega"],
            "entropy": result["entropy"],
            "collapse_risk": result["collapse_risk"],
            "kappa_mean": result["theory_kappa_mean"],
            "kappa_std": result.get("theory_sigma", 0.0),
            "n_samples": result["n_test"],
        },
    )

    gate_results = {}
    for config_name, gk in gatekeepers.items():
        gr = gk.evaluate(cert_input)
        g3_ev = gr.gate_evidence.get("gate_3_admissibility", {})
        kappa_req = float(g3_ev["kappa_req"]) if "kappa_req" in g3_ev else None
        margin = (
            float(result["theory_kappa"] - kappa_req)
            if kappa_req is not None
            else None
        )
        gate_results[config_name] = {
            "gate_decision": str(gr.decision),
            "gate_reached": str(gr.gate_reached),
            "kappa_req": kappa_req,
            "margin": margin,
            "gate_outcomes": {str(k): v for k, v in gr.gate_outcomes.items()},
            "failure_reason": gr.failure_reason,
        }

    return gate_results


# ---------------------------------------------------------------------------
# One (target, solver, fold) group -- certify all embeddings together
# ---------------------------------------------------------------------------

def _certify_and_gate_group(
    embeddings_dict: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    solver_name: str,
    fold_idx: int,
    target: str,
    gatekeepers: dict,
    seed: int,
) -> list:
    """Certify one (target, solver, fold) via theory kappa, then gate both ways."""
    from shared.theory_certifier import certify_group

    t_start = time.time()

    cert_results = certify_group(
        embeddings_dict=embeddings_dict,
        y_train=y_train,
        y_test=y_test,
        solver_name=solver_name,
        seed=seed,
        shared_boundaries=True,  # Fix 3: pooled tercile calibration (ADR-034 FC-1/FC-2)
    )

    wall_clock = round(time.time() - t_start, 2)

    out = []
    for result in cert_results:
        gate_results = _evaluate_gates(result, gatekeepers)
        result["gate_flat"] = gate_results["flat"]
        result["gate_oobleck"] = gate_results["oobleck"]
        result["fold"] = fold_idx
        result["target"] = target
        result["solver"] = solver_name
        result["wall_clock_s"] = wall_clock
        out.append(result)

    return out


def _strip_underscore_keys(data):
    """Remove _ prefixed keys (non-serializable arrays) before JSON dump."""
    if isinstance(data, list):
        return [_strip_underscore_keys(item) for item in data]
    if isinstance(data, dict):
        return {
            k: _strip_underscore_keys(v)
            for k, v in data.items()
            if not k.startswith("_")
        }
    return data


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(data_dir: str, repr_dir: str, output_dir: str):
    """Run full S019A experiment with theory kappa + paired flat/Oobleck gates."""
    import pandas as pd
    from joblib import Parallel, delayed

    log.info("S019A: Certificate Invariance Gradient (theory kappa D*/D)")
    log.info("  Targets: %s", S019A_TARGETS)
    log.info("  Embeddings: %s", EMBEDDINGS)
    log.info("  Solvers: %s", SOLVERS)
    log.info("  Folds: %d, Seed: %d", N_FOLDS, SEED)
    log.info("  Paired eval: flat (lambda=0) vs oobleck (ADR-023 sigmoidal)")

    t0 = time.time()

    # Load data
    data_path = Path(data_dir)
    # Try geocert v23.0.2 first, fall back to legacy name
    feat_path = data_path / "georsct_table.parquet"
    if not feat_path.exists():
        feat_path = data_path / "zcta_features_labels.parquet"
    df = pd.read_parquet(feat_path)
    log.info("Loaded %d ZCTAs", len(df))

    # County groups for GroupKFold
    county_col = next(
        (c for c in ("county_fips", "county_name", "state_fips", "state")
         if c in df.columns), None
    )
    if county_col is None:
        raise KeyError("No county/state grouping column found in data")
    groups = df[county_col].values

    # ACS feature columns
    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
    )

    # Build embedding representations
    repr_path = Path(repr_dir) if repr_dir else None
    embeddings = _build_embeddings(df, acs_cols, repr_path)

    # Paired gatekeepers
    gatekeepers = _make_gatekeepers()
    log.info(
        "  Flat config: kappa_base=%.2f, lambda=0",
        gatekeepers["flat"].config.kappa_base,
    )
    log.info(
        "  Oobleck config: kappa_base=%.2f, delta=%.1f, steepness=%.1f, sigma_c=%.2f",
        gatekeepers["oobleck"].config.kappa_base,
        gatekeepers["oobleck"].config.lambda_turbulence,
        gatekeepers["oobleck"].config.steepness,
        gatekeepers["oobleck"].config.sigma_c,
    )

    # Results collector
    all_results = []
    failed_targets = []

    gkf = GroupKFold(n_splits=N_FOLDS)
    n_cpus = os.cpu_count() or 1
    log.info("Using %d CPUs for parallel fits", n_cpus)

    # Resume: check for previously completed targets
    completed = _load_completed_targets()
    for target in completed & set(S019A_TARGETS):
        restored = _load_checkpoint(target)
        if restored:
            all_results.extend(restored)
            log.info("  Restored %d rows for %s from checkpoint", len(restored), target)

    for target in S019A_TARGETS:
        if target in completed:
            log.info("  %s: SKIP (already checkpointed)", target)
            continue

        # Resolve column name (parquet uses 'target_' prefix)
        target_col = f"target_{target}" if f"target_{target}" in df.columns else target
        y = df[target_col].values.astype(np.float64)
        nan_mask = ~np.isnan(y)

        # Cache fold indices once per target (same for all embeddings/solvers)
        y_masked = y[nan_mask]
        groups_masked = groups[nan_mask]
        folds = list(gkf.split(np.zeros(nan_mask.sum()), y_masked, groups_masked))

        try:
            # Parallelize over (solver, fold) pairs since certify_group
            # processes all 7 embeddings together for LOO theory kappa
            def _run_group(solver_name, fold_idx, train_idx, test_idx):
                emb_dict = {}
                for emb_name in EMBEDDINGS:
                    Z_masked = embeddings[emb_name][nan_mask]
                    emb_dict[emb_name] = {
                        "train": Z_masked[train_idx],
                        "test": Z_masked[test_idx],
                    }
                return _certify_and_gate_group(
                    embeddings_dict=emb_dict,
                    y_train=y_masked[train_idx],
                    y_test=y_masked[test_idx],
                    solver_name=solver_name,
                    fold_idx=fold_idx,
                    target=target,
                    gatekeepers=gatekeepers,
                    seed=SEED + fold_idx,
                )

            jobs = []
            for solver_name in SOLVERS:
                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    jobs.append(delayed(_run_group)(
                        solver_name, fold_idx, train_idx, test_idx,
                    ))

            group_results = Parallel(n_jobs=n_cpus, prefer="processes")(jobs)

            # Flatten: each group returns a list of embedding results
            results = []
            for group in group_results:
                results.extend(group)

            # Log per-cell summaries (theory kappa + flat vs oobleck)
            for emb_name in EMBEDDINGS:
                for solver_name in SOLVERS:
                    cell = [r for r in results
                            if r["embedding"] == emb_name and r["solver"] == solver_name]
                    if not cell:
                        continue
                    mean_r2 = np.mean([r["r2"] for r in cell])
                    mean_sigma = np.mean([r["sigma"] for r in cell])
                    mean_tk = np.mean([r["theory_kappa"] for r in cell])
                    mean_pk = np.mean([r["proxy_kappa"] for r in cell])
                    flat_decisions = [r["gate_flat"]["gate_decision"] for r in cell]
                    oob_decisions = [r["gate_oobleck"]["gate_decision"] for r in cell]
                    flat_pass = sum(1 for d in flat_decisions if "EXECUTE" in d)
                    oob_pass = sum(1 for d in oob_decisions if "EXECUTE" in d)
                    log.info(
                        "  %s x %s x %s: R2=%.3f sigma=%.3f "
                        "tk=%.3f pk=%.3f flat=%d/%d oobleck=%d/%d",
                        target, emb_name, solver_name, mean_r2, mean_sigma,
                        mean_tk, mean_pk, flat_pass, len(cell), oob_pass, len(cell),
                    )

            all_results.extend(results)

            # Checkpoint to S3
            _save_checkpoint(target, _strip_underscore_keys(results))

        except Exception as e:
            log.error("  FAILED %s: %s", target, e)
            log.error("  %s", traceback.format_exc())
            failed_targets.append(target)
            continue

    if failed_targets:
        log.warning("FAILED TARGETS (%d): %s", len(failed_targets), failed_targets)

    elapsed = time.time() - t0
    log.info("S019A complete: %d results in %.1f seconds", len(all_results), elapsed)

    # Summary statistics
    if all_results:
        _log_summary(all_results)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clean_results = _strip_underscore_keys(all_results)
    with open(out_path / "s019a_results.json", "w") as f:
        json.dump(clean_results, f, indent=2, default=str)

    # Save per-sample arrays as npz (proxy kappa, theory kappa, probs per fold)
    per_sample = {}
    for r in all_results:
        key = f"{r['target']}_{r['embedding']}_{r['solver']}_fold{r['fold']}"
        if r.get("_kappa_per_proxy") is not None:
            per_sample[f"{key}_kappa_proxy"] = r["_kappa_per_proxy"]
        if r.get("_kappa_per_theory") is not None:
            per_sample[f"{key}_kappa_theory"] = r["_kappa_per_theory"]
        if r.get("_probs_test") is not None:
            per_sample[f"{key}_probs"] = r["_probs_test"]

    np.savez_compressed(out_path / "s019a_per_sample.npz", **per_sample)

    log.info("Results saved to %s", out_path)
    return all_results


def _log_summary(results: list):
    """Log hypothesis-relevant summary stats across all targets."""
    from scipy import stats as sp_stats

    sigmas = np.array([r["sigma"] for r in results])

    log.info("--- S019A Summary ---")
    log.info("  Total cells: %d", len(results))
    log.info("  sigma range: [%.3f, %.3f], mean=%.3f",
             sigmas.min(), sigmas.max(), sigmas.mean())

    # Theory vs proxy kappa comparison
    theory_kappas = np.array([r["theory_kappa"] for r in results])
    proxy_kappas = np.array([
        r.get("proxy_kappa", r.get("kappa", 0.0)) for r in results
    ])
    log.info("  theory_kappa: mean=%.3f, std=%.3f",
             theory_kappas.mean(), theory_kappas.std())
    log.info("  proxy_kappa:  mean=%.3f, std=%.3f",
             proxy_kappas.mean(), proxy_kappas.std())

    # Spearman(margin, R2) under oobleck
    margins_oob = []
    r2s_valid = []
    for r in results:
        gate_oob = r.get("gate_oobleck", {})
        m = gate_oob.get("margin")
        if m is not None:
            margins_oob.append(m)
            r2s_valid.append(r["r2"])
    margins_oob = np.array(margins_oob)
    r2s_valid = np.array(r2s_valid)

    if len(margins_oob) > 5:
        rho, p = sp_stats.spearmanr(margins_oob, r2s_valid)
        log.info(
            "  Spearman(margin_oobleck, R2) = %.3f (p=%.4f)", rho, p,
        )

    # Rejection rates for high-N cells
    high_n = [r for r in results if r["N"] > 0.35]
    if high_n:
        flat_reject = sum(
            1 for r in high_n
            if "EXECUTE" not in r.get("gate_flat", {}).get("gate_decision", "")
        )
        oob_reject = sum(
            1 for r in high_n
            if "EXECUTE" not in r.get("gate_oobleck", {}).get("gate_decision", "")
        )
        log.info(
            "  High-N cells (%d): flat_reject=%d, oobleck_reject=%d",
            len(high_n), flat_reject, oob_reject,
        )

    # kappa_req delta for low-N cells (should be near zero)
    low_n = [r for r in results if r["N"] < 0.2]
    if low_n:
        deltas = [
            abs(
                (r.get("gate_oobleck", {}).get("kappa_req") or 0)
                - (r.get("gate_flat", {}).get("kappa_req") or 0)
            )
            for r in low_n
            if r.get("gate_oobleck", {}).get("kappa_req") is not None
            and r.get("gate_flat", {}).get("kappa_req") is not None
        ]
        if deltas:
            log.info(
                "  Low-N cells (%d): mean |kappa_req delta| = %.4f",
                len(low_n), np.mean(deltas),
            )

    # Per-embedding summary
    emb_names = sorted(set(r["embedding"] for r in results))
    for emb in emb_names:
        emb_results = [r for r in results if r["embedding"] == emb]
        mean_r2 = np.mean([r["r2"] for r in emb_results])
        mean_tk = np.mean([r["theory_kappa"] for r in emb_results])
        flat_pass = sum(
            1 for r in emb_results
            if "EXECUTE" in r.get("gate_flat", {}).get("gate_decision", "")
        )
        oob_pass = sum(
            1 for r in emb_results
            if "EXECUTE" in r.get("gate_oobleck", {}).get("gate_decision", "")
        )
        log.info(
            "  %s: R2=%.3f tk=%.3f flat=%d/%d oobleck=%d/%d",
            emb, mean_r2, mean_tk, flat_pass, len(emb_results),
            oob_pass, len(emb_results),
        )


def _build_embeddings(df, acs_cols, repr_path):
    """Build all 7 embedding representations."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_acs = df[acs_cols].values.astype(np.float64)

    # NaN imputation (median)
    medians = np.nanmedian(X_acs, axis=0)
    for j in range(X_acs.shape[1]):
        nans = np.isnan(X_acs[:, j])
        if nans.any():
            X_acs[nans, j] = medians[j]

    embeddings = {}

    # PCA32
    if repr_path and (repr_path / "pca32_v1.npz").exists():
        art = np.load(repr_path / "pca32_v1.npz", allow_pickle=True)
        scaler_mean = art["scaler_mean"]
        scaler_scale = art["scaler_scale"]
        pca_components = art["pca_components"]
        pca_mean = art["pca_mean"]
        stored_schema = list(art["feature_schema"])
        if len(stored_schema) != len(acs_cols):
            col_idx = [acs_cols.index(c) for c in stored_schema]
            X_pca = X_acs[:, col_idx]
        else:
            X_pca = X_acs
        X_scaled = (X_pca - scaler_mean) / scaler_scale
        embeddings["pca_v1"] = (X_scaled - pca_mean) @ pca_components.T
        log.info("PCA32: loaded from stored artifact")
    else:
        scaler = StandardScaler().fit(X_acs)
        X_scaled = scaler.transform(X_acs)
        pca = PCA(n_components=32, random_state=SEED).fit(X_scaled)
        embeddings["pca_v1"] = pca.transform(X_scaled)
        log.info("PCA32: fitted locally (no stored artifact)")

    # Spatial Lag
    if repr_path and (repr_path / "spatial_lag_v1.npz").exists():
        art = np.load(repr_path / "spatial_lag_v1.npz", allow_pickle=True)
        stored_schema = [str(s) for s in art["feature_schema"]]
        # Gather columns matching stored schema (acs + lag + enrich)
        X_lag_cols = []
        for c in stored_schema:
            if c in df.columns:
                X_lag_cols.append(c)
            else:
                log.warning("Spatial lag schema col %s missing, zero-filling", c)
        X_lag = np.zeros((len(df), len(stored_schema)), dtype=np.float64)
        for i, c in enumerate(stored_schema):
            if c in df.columns:
                X_lag[:, i] = df[c].values.astype(np.float64)
        # NaN imputation
        med_lag = np.nanmedian(X_lag, axis=0)
        for j in range(X_lag.shape[1]):
            nans = np.isnan(X_lag[:, j])
            if nans.any():
                X_lag[nans, j] = med_lag[j]
        X_scaled_lag = (X_lag - art["scaler_mean"]) / art["scaler_scale"]
        embeddings["spatial_lag_v1"] = (X_scaled_lag - art["pca_mean"]) @ art["pca_components"].T
        log.info("Spatial Lag: loaded from stored artifact (%d features -> 32d)", len(stored_schema))
    else:
        # Fallback: use all features including lag columns
        lag_cols = sorted(
            c for c in df.columns
            if c.startswith("lag_") and pd.api.types.is_numeric_dtype(df[c])
        )
        enrich_cols = sorted(
            c for c in df.columns
            if c.startswith("enrich_") and pd.api.types.is_numeric_dtype(df[c])
        )
        all_cols = acs_cols + lag_cols + enrich_cols
        X_all = df[all_cols].values.astype(np.float64)
        med_all = np.nanmedian(X_all, axis=0)
        for j in range(X_all.shape[1]):
            nans = np.isnan(X_all[:, j])
            if nans.any():
                X_all[nans, j] = med_all[j]
        scaler_all = StandardScaler().fit(X_all)
        pca_all = PCA(n_components=32, random_state=SEED).fit(scaler_all.transform(X_all))
        embeddings["spatial_lag_v1"] = pca_all.transform(scaler_all.transform(X_all))
        log.info("Spatial Lag: fitted locally from %d features", len(all_cols))

    # Raw ACS baseline — no dimensionality reduction, just scaled features
    scaler_raw = StandardScaler().fit(X_acs)
    embeddings["acs_raw"] = scaler_raw.transform(X_acs)
    log.info("ACS raw: %d features (no PCA)", X_acs.shape[1])

    # Geospatial arm — lat/lon + flood + HIFLD healthcare + SVI + drive times
    geo_cols = sorted(
        c for c in df.columns
        if c.startswith(("flood_", "hifld_", "svi_", "drive_min_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ) + ["latitude", "longitude"]
    geo_cols = [c for c in geo_cols if c in df.columns]
    X_geo = df[geo_cols].values.astype(np.float64)
    med_geo = np.nanmedian(X_geo, axis=0)
    for j in range(X_geo.shape[1]):
        nans = np.isnan(X_geo[:, j])
        if nans.any():
            X_geo[nans, j] = med_geo[j]
    scaler_geo = StandardScaler().fit(X_geo)
    embeddings["geo_spatial"] = scaler_geo.transform(X_geo)
    log.info("Geospatial: %d features (lat/lon, flood, HIFLD, SVI, drive)", len(geo_cols))

    # Domain features — ACS + all enrichment columns (SVI, flood, HIFLD, drive, lag), scaled, no PCA
    domain_cols = sorted(
        c for c in df.columns
        if c.startswith(("acs_", "lag_", "svi_", "flood_", "hifld_", "drive_min_"))
        and pd.api.types.is_numeric_dtype(df[c])
    )
    X_domain = df[domain_cols].values.astype(np.float64)
    for j in range(X_domain.shape[1]):
        nans = np.isnan(X_domain[:, j])
        if nans.any():
            X_domain[nans, j] = np.nanmedian(X_domain[:, j])
    scaler_domain = StandardScaler().fit(X_domain)
    embeddings["domain_features"] = scaler_domain.transform(X_domain)
    log.info("Domain features: %d columns (ACS + lag + SVI + flood + HIFLD + drive)", len(domain_cols))

    # Noisy control — PCA32 + Gaussian noise, calibrates the floor
    rng = np.random.RandomState(SEED)
    embeddings["noisy_control"] = embeddings["pca_v1"] + rng.normal(0, 1.0, embeddings["pca_v1"].shape)
    log.info("Noisy control: PCA32 + N(0,1) noise")

    # GNN latents (pre-computed, stored as latent array)
    # Canonical format: npz must have 'zcta_id' key; df must have 'zcta_id' column.
    # No raw-order fallback -- misalignment is catastrophic (R2 ~ 0), never silent.
    gnn_candidates = ["zcta_latents_v1.npz", "gnn_v2_latents.npz"]
    gnn_loaded = False
    if repr_path:
        for gnn_file in gnn_candidates:
            if (repr_path / gnn_file).exists():
                art = np.load(repr_path / gnn_file, allow_pickle=True)
                # Canonical latent key
                if "latents" in art:
                    Z_raw = art["latents"]
                elif "Z" in art:
                    Z_raw = art["Z"]
                else:
                    raise KeyError(
                        f"GNN {gnn_file}: no 'latents' or 'Z' key; found {list(art.keys())}"
                    )
                # Canonical ID key -- must be 'zcta_id', no fallback
                if "zcta_id" not in art:
                    raise KeyError(
                        f"GNN {gnn_file}: missing canonical 'zcta_id' key; "
                        f"found {list(art.keys())}. "
                        f"Re-export the artifact with zcta_id as the row-ID array."
                    )
                gnn_ids = np.array([str(x) for x in art["zcta_id"]])
                # df must have 'zcta_id' column -- no raw-order fallback ever
                if "zcta_id" not in df.columns:
                    raise KeyError(
                        f"DataFrame missing 'zcta_id' column; cannot align GNN latents. "
                        f"Found: {list(df.columns[:10])}"
                    )
                df_ids = df["zcta_id"].astype(str).values
                if len(gnn_ids) != len(df_ids):
                    raise ValueError(
                        f"GNN zcta_id length mismatch: npz has {len(gnn_ids)}, "
                        f"df has {len(df_ids)}"
                    )
                gnn_id_to_idx = {zid: i for i, zid in enumerate(gnn_ids)}
                missing = [zid for zid in df_ids if zid not in gnn_id_to_idx]
                if missing:
                    raise ValueError(
                        f"GNN: {len(missing)} df zcta_ids not found in npz "
                        f"(first 5: {missing[:5]})"
                    )
                reindex = np.array([gnn_id_to_idx[zid] for zid in df_ids])
                embeddings["gnn_v2"] = Z_raw[reindex]
                n_already = int(np.sum(reindex == np.arange(len(reindex))))
                log.info("GNN: loaded from %s (shape=%s), "
                         "reindexed by zcta_id (%d/%d already in order)",
                         gnn_file, Z_raw.shape, n_already, len(df_ids))
                gnn_loaded = True
                break
    if not gnn_loaded:
        raise FileNotFoundError(
            f"GNN latents not found in repr_path={repr_path}. "
            f"Expected one of: {gnn_candidates}"
        )

    return embeddings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S019A Certificate Invariance Gradient")
    parser.add_argument("--data-dir", required=True, help="Path to CONUS-27 data")
    parser.add_argument("--repr-dir", default=None, help="Path to representation artifacts")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: config valid")
        log.info("  data_dir: %s", args.data_dir)
        log.info("  repr_dir: %s", args.repr_dir)
        log.info("  output_dir: %s", args.output_dir)
        log.info("  cells: %d", len(S019A_TARGETS) * len(EMBEDDINGS) * len(SOLVERS))
        log.info("  fits: %d (x2 gate configs)",
                 len(S019A_TARGETS) * len(EMBEDDINGS) * len(SOLVERS) * N_FOLDS)
        log.info("  kappa: theory D*/D (RegressionKappaEvaluator, LOO)")
        log.info("  parallelism: over (solver, fold) pairs, %d groups per target",
                 len(SOLVERS) * N_FOLDS)
        gatekeepers = _make_gatekeepers()
        log.info(
            "  flat: kappa_base=%.2f, lambda=%.1f",
            gatekeepers["flat"].config.kappa_base,
            gatekeepers["flat"].config.lambda_turbulence,
        )
        log.info(
            "  oobleck: kappa_base=%.2f, delta=%.1f, steepness=%.1f, sigma_c=%.2f",
            gatekeepers["oobleck"].config.kappa_base,
            gatekeepers["oobleck"].config.lambda_turbulence,
            gatekeepers["oobleck"].config.steepness,
            gatekeepers["oobleck"].config.sigma_c,
        )
        sys.exit(0)

    run_experiment(args.data_dir, args.repr_dir, args.output_dir)
