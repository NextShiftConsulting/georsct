#!/usr/bin/env python3
"""
train_and_export_v2.py — geo_cert v2: PCA-32 Champion + Discovered Structure.

v2 appends features surfaced during allocator analysis (S -> R transfer):
  - latitude, longitude           (spatial gradient, from Q5: Connection)
  - log_population                (scale signal, from Q1: Scope)
  - 31 lag_acs_* neighbor-means   (spatial structure, from Q5: Connection)

Total: 31 ACS + 3 spatial/scale + 31 lag = 65 features -> PCA-32 -> GBDT

Reproducibility guarantees:
  - Same splits as v1 (split_imputation column)
  - Same GBDT hyperparameters (n_estimators=200, max_depth=4, lr=0.05)
  - Same PCA dimensionality (32 components)
  - Timestamped results artifact (immutable) alongside serving artifact (mutable)
  - Per-task R² delta vs v1 baseline for direct comparison

Usage:
  python train_and_export_v2.py --dry-run
  python train_and_export_v2.py --local-data C:/tmp/geo_data
  python train_and_export_v2.py --compare-v1   # prints per-task delta vs v1
"""

import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler

from apps.geo_cert.validation import run_spatial_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REGION = "us-east-1"
DATA_BUCKET = "yrsn-datasets"
DATA_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
ADJACENCY_KEY = "rsct_curriculum/series_018/processed/zcta_adjacency.parquet"
OUTPUT_BUCKET = "yrsn-checkpoints"
OUTPUT_KEY = "geo-cert/geo_cert_pca32_v2.json"
MANIFEST_KEY = "geo-cert/manifest_pca32_v2.json"
ENGINES_KEY = "engines.json"

VERSION = "2.0.0"
MODEL_KEY = "geo_pca32_structure_v2"
ENGINE_ID = "geo_cert_pca32_v2"

# v1 baseline R² for delta comparison (from v1 training output)
V1_TASK_R2 = {
    "annual_checkup": 0.55, "arthritis": 0.70, "asthma": 0.62,
    "binge_drinking": 0.46, "bp_medicated": 0.57, "cancer": 0.60,
    "cholesterol_screening": 0.52, "chronic_kidney_disease": 0.72,
    "copd": 0.76, "coronary_heart_disease": 0.73, "dental_visit": 0.58,
    "diabetes": 0.78, "high_blood_pressure": 0.72, "high_cholesterol": 0.51,
    "home_value": 0.68, "income": 0.68, "mental_health_not_good": 0.71,
    "night_lights": 0.85, "obesity": 0.75, "physical_health_not_good": 0.81,
    "physical_inactivity": 0.80, "population_density": 0.80,
    "sleep_less_7hr": 0.68, "smoking": 0.81, "stroke": 0.70,
    "tree_cover": 0.54, "elevation": 0.65,
}

# Task configuration (from S018)
SUPPORTED_TASK_NAMES = [
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
    "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
    "coronary_heart_disease", "dental_visit", "diabetes", "elevation",
    "high_blood_pressure", "high_cholesterol", "home_value", "income",
    "mental_health_not_good", "night_lights", "obesity",
    "physical_health_not_good", "physical_inactivity", "population_density",
    "sleep_less_7hr", "smoking", "stroke", "tree_cover",
]

HEALTH_TASKS = {
    "arthritis", "asthma", "bp_medicated", "cancer", "chronic_kidney_disease",
    "copd", "coronary_heart_disease", "diabetes", "high_blood_pressure",
    "high_cholesterol", "mental_health_not_good", "obesity",
    "physical_health_not_good", "physical_inactivity", "smoking", "stroke",
}

UNSUPPORTED_TASKS = [
    "binge_drinking", "cholesterol_screening", "dental_visit",
    "income", "sleep_less_7hr",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_s3(bucket: str, key: str, suffix: str = ".parquet") -> str:
    """Download S3 object to a temp file. Returns local path."""
    s3 = boto3.client("s3", region_name=REGION)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    log.info(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, tmp.name)
    tmp.close()
    return tmp.name


def load_data(local_path: str = None) -> pd.DataFrame:
    """Load ZCTA features+labels from local path or S3."""
    if local_path:
        p = Path(local_path)
        if p.is_dir():
            candidate = p / "zcta_features_labels.parquet"
            if candidate.exists():
                log.info(f"Loading features from {candidate}")
                return pd.read_parquet(candidate)
        elif p.exists():
            log.info(f"Loading features from {p}")
            return pd.read_parquet(p)

    tmp_path = _download_s3(DATA_BUCKET, DATA_KEY)
    df = pd.read_parquet(tmp_path)
    os.unlink(tmp_path)
    return df


def load_adjacency(local_path: str = None) -> dict:
    """Load adjacency as {zcta_id: [neighbor_zcta_ids]}."""
    if local_path:
        p = Path(local_path)
        if p.is_dir():
            for fname in ("zcta_adjacency.parquet", "zcta_adjacency.json"):
                candidate = p / fname
                if candidate.exists():
                    return _parse_adjacency(candidate)
        elif p.exists():
            return _parse_adjacency(p)

    tmp_path = _download_s3(DATA_BUCKET, ADJACENCY_KEY)
    result = _parse_adjacency(Path(tmp_path))
    os.unlink(tmp_path)
    return result


def _parse_adjacency(path: Path) -> dict:
    """Parse adjacency file into {zcta: [neighbors]}."""
    log.info(f"Loading adjacency from {path}")
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    adj_df = pd.read_parquet(path)
    zcta_col = "zcta" if "zcta" in adj_df.columns else "zcta_id"
    nbr_col = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"
    result: dict = {}
    for row in adj_df.itertuples(index=False):
        z = str(getattr(row, zcta_col)).zfill(5)
        nb = str(getattr(row, nbr_col)).zfill(5)
        result.setdefault(z, []).append(nb)
    log.info(f"Adjacency: {len(result)} ZCTAs with neighbors")
    return result


# ---------------------------------------------------------------------------
# Feature engineering — discovered structure (S -> R transfer)
# ---------------------------------------------------------------------------

def compute_spatial_lags(
    df: pd.DataFrame,
    feature_cols: list,
    adjacency: dict,
    id_col: str = "zcta_id",
) -> pd.DataFrame:
    """Compute mean-neighbor features for each ZCTA."""
    id_to_idx = {str(v).zfill(5): i for i, v in enumerate(df[id_col].values)}
    X = df[feature_cols].values.astype(np.float64)

    # Impute NaN with column medians before neighbor averaging
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = col_medians[j]

    lag_X = np.zeros_like(X)
    hit, miss = 0, 0
    for i, zcta in enumerate(df[id_col].values):
        neighbors = adjacency.get(str(zcta).zfill(5), [])
        valid = [id_to_idx[nb] for nb in neighbors if nb in id_to_idx]
        if valid:
            lag_X[i] = X[valid].mean(axis=0)
            hit += 1
        else:
            miss += 1

    log.info(f"Spatial lag: {hit} ZCTAs with neighbors, {miss} without (zero lag)")
    lag_cols = {f"lag_{c}": lag_X[:, j] for j, c in enumerate(feature_cols)}
    return df.assign(**lag_cols)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discovered-structure features: latitude, longitude, log_population."""
    df = df.copy()
    # Log population — captures nonlinear scale effects (Q1: Scope)
    pop = df["population"].values.astype(np.float64)
    df["struct_log_population"] = np.log1p(np.maximum(pop, 0))
    # Spatial coordinates (Q5: Connection)
    df["struct_latitude"] = df["latitude"].astype(np.float64)
    df["struct_longitude"] = df["longitude"].astype(np.float64)
    log.info("Added structure features: struct_log_population, struct_latitude, struct_longitude")
    return df


def get_leakage_safe_cols(feature_cols: list, task: str) -> list:
    """Remove leakage-prone features per task (handles acs_, lag_, struct_ prefixes)."""
    if task in HEALTH_TASKS:
        return [c for c in feature_cols
                if c != "acs_pct_no_insurance" and c != "lag_acs_pct_no_insurance"]
    if task in ("home_value", "income"):
        return [c for c in feature_cols
                if "income" not in c.lower() and "poverty" not in c.lower()]
    return list(feature_cols)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame) -> dict:
    """Train PCA-32 on [ACS + structure + lag] + per-task GBDTs + proxy certifiers."""
    # Split: same as v1 (split_imputation column, geographic)
    if "split_imputation" in df.columns:
        test_mask = (df["split_imputation"] == "test").values
        train_mask = ~test_mask
        log.info(f"Split (split_imputation): train={train_mask.sum()}, test={test_mask.sum()}")
    else:
        rng = np.random.RandomState(42)
        train_mask = rng.rand(len(df)) < 0.8
        test_mask = ~train_mask
        log.info(f"Split (random): train={train_mask.sum()}, test={test_mask.sum()}")

    # Feature columns: acs_* + lag_acs_* + struct_*
    skip = {
        "state", "latitude", "longitude", "zcta_id", "county_name", "population",
        "split_imputation", "split_extrap", "split_superres",
    }
    feature_cols = sorted(
        c for c in df.columns
        if (c.startswith("acs_") or c.startswith("lag_acs_") or c.startswith("struct_"))
        and c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
    )
    n_acs = sum(1 for c in feature_cols if c.startswith("acs_"))
    n_lag = sum(1 for c in feature_cols if c.startswith("lag_"))
    n_struct = sum(1 for c in feature_cols if c.startswith("struct_"))
    log.info(f"Feature columns: {len(feature_cols)} ({n_acs} ACS + {n_struct} structure + {n_lag} spatial lags)")

    # Fit PCA-32 (health baseline — no insurance column)
    safe_cols = get_leakage_safe_cols(feature_cols, "diabetes")
    X = df[safe_cols].values.astype(np.float64)
    medians = np.nanmedian(X[train_mask], axis=0)
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = medians[j]

    n_components = min(32, X.shape[1])
    pipe = SKPipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])
    pipe.fit(X[train_mask])
    z_all = pipe.transform(X)  # float64 — no downcasting (v1 lesson)
    log.info(f"PCA-32: {X.shape[1]} features -> {n_components} components")
    log.info(f"  Explained variance: {pipe.named_steps['pca'].explained_variance_ratio_.sum():.3f}")

    # Train per-task GBDTs (same hyperparams as v1)
    task_models = {}
    certifier_config = {}
    task_results = []

    for task in SUPPORTED_TASK_NAMES:
        target_col = f"target_{task}"
        if target_col not in df.columns:
            log.warning(f"  SKIP {task}: column '{target_col}' not found")
            continue

        y = df[target_col].values.astype(np.float64)
        valid = ~np.isnan(y)
        tr = train_mask & valid
        te = test_mask & valid

        if tr.sum() < 100:
            log.warning(f"  SKIP {task}: too few training samples ({tr.sum()})")
            continue

        z_tr, y_tr = z_all[tr], y[tr]
        z_te, y_te = z_all[te], y[te]

        # Same hyperparameters as v1 — isolate the feature-set effect
        gbdt = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
        )
        gbdt.fit(z_tr, y_tr)
        y_pred = gbdt.predict(z_te)
        r2 = r2_score(y_te, y_pred)

        task_models[task] = gbdt

        # Proxy certifier (Spearman-weighted kappa)
        n_dims = z_tr.shape[1]
        weights = np.zeros(n_dims, dtype=np.float32)
        for j in range(n_dims):
            rho, _ = spearmanr(z_tr[:, j], y_tr)
            weights[j] = -rho if not np.isnan(rho) else 0.0
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights /= norm
        raw_scores = z_tr @ weights
        kappas = 1.0 / (1.0 + np.exp(-raw_scores))
        edges = [float(np.quantile(kappas, q)) for q in [0.2, 0.4, 0.6, 0.8]]

        certifier_config[task] = {
            "feature_weights": weights.tolist(),
            "bucket_edges": edges,
            "sigma_scale": float(1.0 / max(np.std(z_tr), 1e-6)),
        }

        # Delta vs v1
        v1_r2 = V1_TASK_R2.get(task, 0.0)
        delta = r2 - v1_r2
        marker = "+" if delta > 0 else ""
        task_results.append({
            "task": task, "r2": round(r2, 4),
            "v1_r2": round(v1_r2, 4), "delta": round(delta, 4),
            "n_train": int(tr.sum()),
        })
        log.info(f"  {task}: R2={r2:.4f} (v1={v1_r2:.4f}, delta={marker}{delta:.4f})")

    log.info(f"\nTrained {len(task_models)} task models")

    # Aggregate comparison
    v2_r2s = [e["r2"] for e in task_results]
    v1_r2s = [e["v1_r2"] for e in task_results]
    log.info(f"  v2 mean R2: {np.mean(v2_r2s):.4f}")
    log.info(f"  v1 mean R2: {np.mean(v1_r2s):.4f}")
    log.info(f"  delta:      {np.mean(v2_r2s) - np.mean(v1_r2s):+.4f}")
    improved = sum(1 for e in task_results if e["delta"] > 0)
    log.info(f"  Improved: {improved}/{len(task_results)} tasks")

    return {
        "pipe": pipe,
        "feature_schema": safe_cols,
        "task_models": task_models,
        "certifier_config": certifier_config,
        "task_results": task_results,
        "z_all": z_all,
        "train_mask": train_mask,
        "feature_breakdown": {"n_acs": n_acs, "n_lag": n_lag, "n_struct": n_struct},
    }


# ---------------------------------------------------------------------------
# Serialization (identical to v1 — same JSON format)
# ---------------------------------------------------------------------------

def serialize_pca(pipe, feature_schema: list) -> dict:
    """Serialize PCA pipeline to JSON-compatible dict."""
    scaler = pipe.named_steps["scaler"]
    pca = pipe.named_steps["pca"]
    return {
        "type": "pca_v2_structure",
        "feature_order": feature_schema,
        "n_features": len(feature_schema),
        "n_components": int(pca.n_components_),
        "means": scaler.mean_.tolist(),
        "stds": scaler.scale_.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _extract_tree(estimator) -> dict:
    """Extract sklearn DecisionTreeRegressor into JSON-walkable structure."""
    t = estimator.tree_

    def _recurse(node_id: int) -> dict:
        if t.children_left[node_id] == -1:
            return {"v": float(t.value[node_id][0, 0])}
        return {
            "f": int(t.feature[node_id]),
            "t": float(t.threshold[node_id]),
            "l": _recurse(int(t.children_left[node_id])),
            "r": _recurse(int(t.children_right[node_id])),
        }

    return _recurse(0)


def serialize_gbdt(model) -> dict:
    """Serialize GradientBoostingRegressor to compact JSON."""
    trees = [_extract_tree(est[0]) for est in model.estimators_]
    return {
        "n": len(trees),
        "lr": float(model.learning_rate),
        "init": float(np.asarray(model.init_.constant_).flat[0]),
        "trees": trees,
    }


# ---------------------------------------------------------------------------
# Fidelity validation
# ---------------------------------------------------------------------------

def _walk(node: dict, x: np.ndarray) -> float:
    """Walk serialized tree. Must match sklearn exactly."""
    if "v" in node:
        return node["v"]
    if x[node["f"]] <= node["t"]:
        return _walk(node["l"], x)
    return _walk(node["r"], x)


def predict_json(gbdt_json: dict, X: np.ndarray) -> np.ndarray:
    """Predict using serialized JSON trees."""
    preds = np.full(len(X), gbdt_json["init"])
    lr = gbdt_json["lr"]
    for tree in gbdt_json["trees"]:
        for i in range(len(X)):
            preds[i] += lr * _walk(tree, X[i])
    return preds


def validate_fidelity(task_models: dict, serialized: dict, z: np.ndarray,
                      n_samples: int = 500, tolerance: float = 1e-6) -> dict:
    """Verify JSON trees reproduce sklearn predictions exactly."""
    log.info("Validating fidelity...")
    idx = np.random.RandomState(99).choice(len(z), min(n_samples, len(z)), replace=False)
    Z = z[idx]

    worst = 0.0
    for task, model in task_models.items():
        native = model.predict(Z)
        json_pred = predict_json(serialized[task], Z)
        diff = float(np.max(np.abs(native - json_pred)))
        worst = max(worst, diff)
        if diff >= tolerance:
            raise ValueError(f"FIDELITY FAIL {task}: max_diff={diff:.2e} > {tolerance:.0e}")

    log.info(f"  Fidelity PASS: worst_diff={worst:.2e} (tol={tolerance:.0e})")
    return {"worst_diff": worst, "n_validated": len(Z), "n_tasks": len(task_models)}


# ---------------------------------------------------------------------------
# Build artifact + manifest
# ---------------------------------------------------------------------------

def build_artifact(trained: dict) -> dict:
    """Build complete geo_cert_v2 artifact."""
    rep = serialize_pca(trained["pipe"], trained["feature_schema"])

    log.info("Serializing GBDTs...")
    predictors = {}
    for task, model in trained["task_models"].items():
        predictors[task] = serialize_gbdt(model)
    log.info(f"  Serialized {len(predictors)} task GBDTs")

    # Validate fidelity
    fidelity = validate_fidelity(
        trained["task_models"], predictors, trained["z_all"]
    )

    # Task metrics with v1 delta
    task_metrics = {}
    for entry in trained["task_results"]:
        task_metrics[entry["task"]] = {
            "r2": entry["r2"],
            "v1_r2": entry["v1_r2"],
            "delta": entry["delta"],
            "n_train": entry["n_train"],
        }

    mean_r2 = np.mean([e["r2"] for e in trained["task_results"]])
    v1_mean = np.mean([e["v1_r2"] for e in trained["task_results"]])
    fb = trained["feature_breakdown"]

    return {
        "format": "geo_cert_v2",
        "version": VERSION,
        "model_key": MODEL_KEY,
        "created": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "parent_model": "geo_cert_pca32_v1",
            "parent_version": "1.0.0",
            "change_type": "feature_expansion",
            "change_summary": (
                f"Appended {fb['n_struct']} structure features (lat, lon, log_pop) "
                f"+ {fb['n_lag']} spatial lag features to {fb['n_acs']} ACS base. "
                f"Same splits, same GBDT hyperparams. S->R transfer from allocator analysis."
            ),
        },
        "representation": rep,
        "predictors": predictors,
        "certifier": trained["certifier_config"],
        "policy": {
            "unsupported_tasks": UNSUPPORTED_TASKS,
            "refuse_kappa_max": 0.35,
            "kappa_base": 0.45,
            "lambda_oobleck": 0.40,
        },
        "metadata": {
            "n_tasks_trained": len(predictors),
            "mean_r2": round(float(mean_r2), 4),
            "v1_mean_r2": round(float(v1_mean), 4),
            "mean_r2_delta": round(float(mean_r2 - v1_mean), 4),
            "task_metrics": task_metrics,
            "fidelity": fidelity,
            "feature_breakdown": fb,
            "inference_deps": "numpy only",
            "sklearn_version": str(__import__("sklearn").__version__),
        },
    }


def build_manifest(artifact_bytes: bytes) -> dict:
    """Build manifest passing ArtifactManifestModel validation."""
    sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1.0",
        "generated_by": "apps/geo_cert/train_and_export_v2.py",
        "generated_at": now,

        "artifact_id": MODEL_KEY,
        "engine_id": ENGINE_ID,
        "encoder_id": "acs_pca32_structure",
        "task_type": "geospatial",
        "version": VERSION,

        "artifact_uri": f"s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}",
        "sha256": sha256,
        "file_format": "json",

        "src_dim": 8,
        "projection_dim": 32,

        "serving_path": "geo_cert_v1",
        "validation_status": "validated",

        "training_run_id": os.environ.get("TRAINING_JOB_NAME"),
        "created_at": now,
        "classifier_source": "geo_gbdt_proxy",

        "canonical_name": "geo_cert_pca32_structure_v2",
        "metadata": {
            "task_family": "geospatial",
            "compatible_endpoints": ["/geo/score"],
            "decomposition_mode": "proxy_spearman_v2",
            "lineage_parent": "geo_cert_pca32_v1",
        },
    }


def validate_manifest(manifest: dict) -> bool:
    """Validate against ArtifactManifestModel."""
    try:
        from yrsn.model_mgmt.manifests import ArtifactManifestModel
        ArtifactManifestModel(**manifest)
        log.info("ArtifactManifestModel validation: PASS")
        return True
    except ImportError:
        log.info("yrsn not installed, running offline validation")
        return _validate_offline(manifest)
    except Exception as e:
        log.error(f"Manifest validation FAILED: {e}")
        return False


def _validate_offline(manifest: dict) -> bool:
    """Offline validation when yrsn not available."""
    required = [
        "artifact_id", "engine_id", "encoder_id", "task_type", "version",
        "artifact_uri", "sha256", "file_format", "src_dim", "serving_path",
        "validation_status", "created_at",
    ]
    missing = [f for f in required if f not in manifest]
    if missing:
        log.error(f"Missing: {missing}")
        return False

    sha = manifest.get("sha256", "")
    if len(sha) != 64:
        log.error(f"sha256 length {len(sha)} != 64")
        return False

    log.info("Offline validation: PASS")
    return True


# ---------------------------------------------------------------------------
# Timestamped results artifact (immutable)
# ---------------------------------------------------------------------------

def build_results_artifact(trained: dict, artifact_sha256: str) -> dict:
    """Build immutable timestamped results for reproducibility."""
    now = datetime.now(timezone.utc)
    fb = trained["feature_breakdown"]
    return {
        "run_id": f"v2_{now.strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": now.isoformat(),
        "model_version": VERSION,
        "engine_id": ENGINE_ID,
        "parent_model": "geo_cert_pca32_v1",
        "artifact_sha256": artifact_sha256,
        "feature_set": {
            "n_acs": fb["n_acs"],
            "n_structure": fb["n_struct"],
            "n_spatial_lag": fb["n_lag"],
            "total": fb["n_acs"] + fb["n_struct"] + fb["n_lag"],
            "pca_components": 32,
            "structure_features_added": [
                "struct_latitude (Q5: Connection)",
                "struct_longitude (Q5: Connection)",
                "struct_log_population (Q1: Scope)",
                "lag_acs_* x 31 (Q5: Connection, spatial neighbor means)",
            ],
        },
        "hyperparameters": {
            "gbdt_n_estimators": 200,
            "gbdt_max_depth": 4,
            "gbdt_learning_rate": 0.05,
            "gbdt_min_samples_leaf": 40,
            "gbdt_random_state": 42,
            "pca_n_components": 32,
            "pca_random_state": 42,
        },
        "split": "split_imputation (geographic, county-level)",
        "per_task_results": trained["task_results"],
        "aggregate": {
            "v2_mean_r2": round(float(np.mean([e["r2"] for e in trained["task_results"]])), 4),
            "v1_mean_r2": round(float(np.mean([e["v1_r2"] for e in trained["task_results"]])), 4),
            "mean_delta": round(float(np.mean([e["delta"] for e in trained["task_results"]])), 4),
            "tasks_improved": sum(1 for e in trained["task_results"] if e["delta"] > 0),
            "tasks_regressed": sum(1 for e in trained["task_results"] if e["delta"] < 0),
            "tasks_unchanged": sum(1 for e in trained["task_results"] if e["delta"] == 0),
            "total_tasks": len(trained["task_results"]),
        },
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(artifact_bytes: bytes, manifest: dict, results: dict) -> None:
    """Upload artifact + manifest + timestamped results to S3."""
    s3 = boto3.client("s3", region_name=REGION)

    # 1. Serving artifact (mutable — latest v2)
    log.info(f"Uploading artifact ({len(artifact_bytes)/1024:.1f} KB) "
             f"to s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=OUTPUT_KEY, Body=artifact_bytes,
                  ContentType="application/json")

    # 2. Manifest
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=MANIFEST_KEY, Body=manifest_bytes,
                  ContentType="application/json")
    log.info(f"Uploaded manifest to s3://{OUTPUT_BUCKET}/{MANIFEST_KEY}")

    # 3. Timestamped results (immutable — write-once)
    results_key = f"geo-cert/runs/{results['run_id']}.results.json"
    results_bytes = json.dumps(results, indent=2).encode()
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=results_key, Body=results_bytes,
                  ContentType="application/json")
    log.info(f"Uploaded results to s3://{OUTPUT_BUCKET}/{results_key}")

    # 4. Verify round-trip integrity
    log.info("Verifying round-trip integrity...")
    resp = s3.get_object(Bucket=OUTPUT_BUCKET, Key=OUTPUT_KEY)
    downloaded = resp["Body"].read()
    actual_sha = hashlib.sha256(downloaded).hexdigest()
    if actual_sha != manifest["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch: expected {manifest['sha256'][:16]}, got {actual_sha[:16]}"
        )
    log.info(f"  SHA256 verified: {actual_sha[:16]}...")


# ---------------------------------------------------------------------------
# Engine registry update
# ---------------------------------------------------------------------------

def build_engine_entry(manifest: dict) -> dict:
    """Build EngineSpec-compatible entry for engines.json."""
    return {
        "provider": "acs_census",
        "model_name": "pca32_structure_gbdt_v2",
        "src_dim": manifest["src_dim"],
        "artifact_uri": manifest["artifact_uri"].replace(f"s3://{OUTPUT_BUCKET}/", ""),
        "onnx_uri": "",
        "artifact_layout": "bundled_v2",
        "encoder_family": "acs_pca32_structure",
        "runtime_profile": "python_local",
        "preferred_artifact_format": "npz",
        "allowed_artifact_formats": ["json"],
        "validation_status": "validated",
        "status": "active",
        "task_family": "geospatial",
        "compatible_endpoints": ["/geo/score"],
        "decomposition_mode": "proxy_spearman_v2",
        "input_contract": "zcta_features_v2",
        "supports_runtime_embedding": False,
        "served_publicly": False,
        "requires_pair_input": False,
        "supports_text_input": False,
        "supports_embedding_input": False,
        "tier_allowlist": [],
        "deprecated": False,
        "metadata": {
            "classifier_source": manifest.get("classifier_source", "geo_gbdt_proxy"),
            "serving_path": manifest.get("serving_path", "geo_cert_v1"),
            "sha256": manifest["sha256"],
            "n_supported_tasks": "21",
            "benchmark": "CONUS-27",
            "lineage_parent": "geo_cert_pca32_v1",
        },
    }


def update_engine_registry(manifest: dict, dry_run: bool = False) -> None:
    """Add geo_cert_pca32_v2 engine to engines.json on S3."""
    engine_id = manifest["engine_id"]
    entry = build_engine_entry(manifest)

    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.get_object(Bucket=OUTPUT_BUCKET, Key=ENGINES_KEY)
    registry = json.loads(resp["Body"].read())

    if engine_id in registry.get("engines", {}):
        log.info(f"Engine '{engine_id}' already in registry -- updating")
    else:
        log.info(f"Adding engine '{engine_id}' to registry")

    registry.setdefault("engines", {})[engine_id] = entry

    try:
        from yrsn.infrastructure.model_artifacts.registry import ModelRegistry
        test_reg = ModelRegistry._from_config(registry)
        spec = test_reg.get_engine(engine_id)
        log.info(f"  ModelRegistry validation: PASS (engine_id={spec.engine_id}, "
                 f"src_dim={spec.src_dim}, status={spec.status})")
    except ImportError:
        log.warning("  yrsn not installed -- skipping ModelRegistry validation")
    except Exception as e:
        log.error(f"  ModelRegistry validation FAILED: {e}")
        raise

    if dry_run:
        log.info(f"  [DRY RUN] Would update s3://{OUTPUT_BUCKET}/{ENGINES_KEY}")
        log.info(f"  Engine entry:\n{json.dumps(entry, indent=2)}")
        return

    body = json.dumps(registry, indent=2, default=str)
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=ENGINES_KEY, Body=body.encode(),
                  ContentType="application/json")
    log.info(f"  Updated s3://{OUTPUT_BUCKET}/{ENGINES_KEY}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="geo_cert v2: PCA-32 + discovered structure (S->R transfer)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Train locally, don't upload")
    parser.add_argument("--local-data", type=str,
                        help="Path to directory or parquet (features + adjacency)")
    parser.add_argument("--output-local", type=str, help="Write artifacts locally")
    parser.add_argument("--skip-registry", action="store_true",
                        help="Skip engine registry update")
    parser.add_argument("--compare-v1", action="store_true",
                        help="Print detailed per-task v1 vs v2 comparison")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("geo_cert v2 -- Train + Export (S -> R Transfer)")
    log.info("=" * 65)
    log.info("Feature expansion: 31 ACS + 3 structure + 31 spatial lags = 65 features")
    log.info("Same splits, same hyperparams as v1 -- isolating feature effect")
    log.info("=" * 65)

    # 1. Load data
    df = load_data(args.local_data)
    log.info(f"Data: {len(df)} rows, {len(df.columns)} columns")

    # 2. Load adjacency for spatial lags
    adjacency = load_adjacency(args.local_data)

    # 3. Compute spatial lag features
    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
    )
    df = compute_spatial_lags(df, acs_cols, adjacency)

    # 4. Add discovered structure features
    df = add_structure_features(df)

    log.info(f"Enriched data: {len(df)} rows, {len(df.columns)} columns")

    # 4b. Spatial representation QA (Priority 2 from representation_qa_principle.md)
    output_local = getattr(args, "output_local", None)
    run_spatial_validation(df, acs_cols, adjacency, output_dir=output_local)

    # 5. Train
    trained = train_model(df)

    # 6. Build artifact
    artifact = build_artifact(trained)
    artifact_bytes = json.dumps(artifact).encode()
    manifest = build_manifest(artifact_bytes)
    results = build_results_artifact(trained, manifest["sha256"])

    log.info(f"\n{'=' * 65}")
    log.info(f"ARTIFACT SUMMARY")
    log.info(f"{'=' * 65}")
    log.info(f"Artifact:   {len(artifact_bytes)/1024:.1f} KB")
    log.info(f"SHA256:     {manifest['sha256'][:16]}...")
    log.info(f"Tasks:      {artifact['metadata']['n_tasks_trained']}")
    log.info(f"v2 mean R2: {artifact['metadata']['mean_r2']}")
    log.info(f"v1 mean R2: {artifact['metadata']['v1_mean_r2']}")
    log.info(f"Delta:      {artifact['metadata']['mean_r2_delta']:+.4f}")
    log.info(f"Features:   {artifact['representation']['n_features']} -> "
             f"{artifact['representation']['n_components']}")

    valid = validate_manifest(manifest)
    if not valid:
        log.error("Manifest validation failed")
        return

    if args.compare_v1:
        print(f"\n{'=' * 65}")
        print(f"PER-TASK v1 vs v2 COMPARISON")
        print(f"{'=' * 65}")
        print(f"{'Task':<30} {'v1 R2':>8} {'v2 R2':>8} {'Delta':>8} {'Result':>8}")
        print(f"{'-' * 65}")
        for entry in sorted(trained["task_results"], key=lambda e: e["delta"], reverse=True):
            marker = "[+]" if entry["delta"] > 0.005 else ("[-]" if entry["delta"] < -0.005 else "[=]")
            print(f"{entry['task']:<30} {entry['v1_r2']:>8.4f} {entry['r2']:>8.4f} "
                  f"{entry['delta']:>+8.4f} {marker:>8}")
        agg = results["aggregate"]
        print(f"{'-' * 65}")
        print(f"{'MEAN':<30} {agg['v1_mean_r2']:>8.4f} {agg['v2_mean_r2']:>8.4f} "
              f"{agg['mean_delta']:>+8.4f}")
        print(f"Improved: {agg['tasks_improved']}/{agg['total_tasks']}  "
              f"Regressed: {agg['tasks_regressed']}/{agg['total_tasks']}")
        print(f"{'=' * 65}")

    if args.output_local:
        out = Path(args.output_local)
        out.mkdir(parents=True, exist_ok=True)
        (out / "geo_cert_pca32_v2.json").write_bytes(artifact_bytes)
        (out / "manifest_pca32_v2.json").write_text(json.dumps(manifest, indent=2))
        (out / f"{results['run_id']}.results.json").write_text(json.dumps(results, indent=2))
        log.info(f"\nWritten to {out}/")
        return

    if args.dry_run:
        log.info("\n[DRY RUN] Would upload to:")
        log.info(f"  s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
        log.info(f"  s3://{OUTPUT_BUCKET}/{MANIFEST_KEY}")
        results_key = f"geo-cert/runs/{results['run_id']}.results.json"
        log.info(f"  s3://{OUTPUT_BUCKET}/{results_key}")
        if not args.skip_registry:
            update_engine_registry(manifest, dry_run=True)
        return

    # Upload artifact + manifest + timestamped results
    upload(artifact_bytes, manifest, results)

    # Register engine
    if not args.skip_registry:
        update_engine_registry(manifest, dry_run=False)

    log.info(f"\n{'=' * 65}")
    log.info("EXPORT COMPLETE")
    log.info(f"  Artifact:  s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
    log.info(f"  Manifest:  s3://{OUTPUT_BUCKET}/{MANIFEST_KEY}")
    log.info(f"  Results:   s3://{OUTPUT_BUCKET}/geo-cert/runs/{results['run_id']}.results.json")
    log.info(f"  Engine:    {manifest['engine_id']}")
    log.info(f"  Registry:  s3://{OUTPUT_BUCKET}/{ENGINES_KEY}")
    log.info(f"  SHA256:    {manifest['sha256'][:16]}...")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
