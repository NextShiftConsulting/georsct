#!/usr/bin/env python3
"""
train_and_export_gnn_v2.py — geo_cert GNN v2: GraphSAGE + Discovered Structure.

v2 expands the GNN input from 33 ACS features to ~65 features:
  - 33 ACS features (raw census)
  - 3 structure features (lat, lon, log_pop) from allocator Q1/Q5 analysis
  - ~31 lag_acs_* neighbor-mean features (spatial structure from Q5)

The GNN already learns spatial structure via message passing, so the spatial lags
are partially redundant here — but they give the FeatureEncoder a pre-computed
spatial signal to work with BEFORE the SAGE convolutions, which may help early
layers converge faster.

Architecture:
  [ACS + struct + lag] (65) -> FeatureEncoder(65->64) -> SAGEConv(64->64) -> SAGEConv(64->32)
  -> 32-dim latents -> per-task GBDT -> proxy certifier

Reproducibility:
  - Same splits as v1 (split_imputation column)
  - Same GBDT hyperparams (n_estimators=200, max_depth=4, lr=0.05)
  - Same GNN hyperparams (hidden=64, latent=32, lr=1e-3, patience=30)
  - Per-task R² delta vs GNN v1 baseline

Usage:
  python train_and_export_gnn_v2.py --dry-run --compare-v1
  python train_and_export_gnn_v2.py --local-features C:/tmp/geo_data/zcta_features_labels.parquet --local-adjacency C:/tmp/geo_data/zcta_adjacency.parquet --output-local C:/tmp/geo_cert_output_gnn_v2
"""

import argparse
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from apps.geo_cert.validation import run_spatial_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REGION = "us-east-1"
DATA_BUCKET = "yrsn-datasets"
DATA_KEY_FEATURES = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
DATA_KEY_ADJACENCY = "rsct_curriculum/series_018/processed/zcta_adjacency.parquet"
OUTPUT_BUCKET = "yrsn-checkpoints"
OUTPUT_KEY = "geo-cert/geo_cert_gnn_v2.json"
MANIFEST_KEY = "geo-cert/manifest_gnn_v2.json"
ENGINES_KEY = "engines.json"

VERSION = "2.0.0"
MODEL_KEY = "geo_graph_structure_v2"
ENGINE_ID = "geo_cert_gnn_v2"

# GNN v1 baseline R² for delta comparison
V1_TASK_R2 = {
    "annual_checkup": 0.52, "arthritis": 0.67, "asthma": 0.59,
    "binge_drinking": 0.43, "bp_medicated": 0.54, "cancer": 0.57,
    "cholesterol_screening": 0.49, "chronic_kidney_disease": 0.69,
    "copd": 0.73, "coronary_heart_disease": 0.70, "dental_visit": 0.55,
    "diabetes": 0.75, "high_blood_pressure": 0.69, "high_cholesterol": 0.48,
    "home_value": 0.65, "income": 0.65, "mental_health_not_good": 0.68,
    "night_lights": 0.82, "obesity": 0.72, "physical_health_not_good": 0.78,
    "physical_inactivity": 0.77, "population_density": 0.77,
    "sleep_less_7hr": 0.65, "smoking": 0.78, "stroke": 0.67,
    "tree_cover": 0.51, "elevation": 0.62,
}

SUPPORTED_TASK_NAMES = [
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
    "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
    "coronary_heart_disease", "dental_visit", "diabetes", "elevation",
    "high_blood_pressure", "high_cholesterol", "home_value", "income",
    "mental_health_not_good", "night_lights", "obesity",
    "physical_health_not_good", "physical_inactivity", "population_density",
    "sleep_less_7hr", "smoking", "stroke", "tree_cover",
]

HEALTH_TASKS = frozenset(
    t for t in SUPPORTED_TASK_NAMES
    if t not in {"elevation", "home_value", "income", "night_lights",
                 "population_density", "tree_cover"}
)

UNSUPPORTED_TASKS = [
    "binge_drinking", "cholesterol_screening", "dental_visit",
    "income", "sleep_less_7hr",
]


# ---------------------------------------------------------------------------
# GraphSAGE architecture (same as v1 — only input dim changes)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class SAGEConv(nn.Module):
        """Single GraphSAGE layer (mean aggregator)."""

        def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim, bias=bias)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            neigh = torch.sparse.mm(adj_norm, x)
            agg = torch.cat([x, neigh], dim=-1)
            return self.linear(agg)

    class GraphSAGE(nn.Module):
        """2-layer GraphSAGE: n_feat -> hidden -> out_dim."""

        def __init__(self, n_feat: int, hidden: int = 64, out_dim: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            self.sage1 = SAGEConv(hidden, hidden)
            self.sage2 = SAGEConv(hidden, out_dim)
            self.norm1 = nn.LayerNorm(hidden)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            h = self.encoder(x)
            h = F.relu(self.norm1(self.sage1(h, adj_norm)))
            h = self.sage2(h, adj_norm)
            return h

    class GNNModel(nn.Module):
        """GraphSAGE backbone + per-task linear heads."""

        def __init__(self, n_feat: int, n_tasks: int, hidden: int = 64, latent: int = 32) -> None:
            super().__init__()
            self.sage = GraphSAGE(n_feat, hidden, latent)
            self.heads = nn.ModuleList([nn.Linear(latent, 1) for _ in range(n_tasks)])

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor):
            z = self.sage(x, adj_norm)
            out = torch.cat([h(z) for h in self.heads], dim=-1)
            return z, out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(local_features: str = None, local_adjacency: str = None):
    """Load training data and adjacency."""
    s3 = boto3.client("s3", region_name=REGION)

    if local_features and Path(local_features).exists():
        log.info(f"Loading features from {local_features}")
        df = pd.read_parquet(local_features)
    else:
        log.info(f"Downloading from s3://{DATA_BUCKET}/{DATA_KEY_FEATURES}")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            s3.download_file(DATA_BUCKET, DATA_KEY_FEATURES, f.name)
            df = pd.read_parquet(f.name)
            os.unlink(f.name)

    if local_adjacency and Path(local_adjacency).exists():
        log.info(f"Loading adjacency from {local_adjacency}")
        adj_df = pd.read_parquet(local_adjacency)
    else:
        log.info(f"Downloading from s3://{DATA_BUCKET}/{DATA_KEY_ADJACENCY}")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            s3.download_file(DATA_BUCKET, DATA_KEY_ADJACENCY, f.name)
            adj_df = pd.read_parquet(f.name)
            os.unlink(f.name)

    log.info(f"Features: {len(df)} rows x {len(df.columns)} cols")
    log.info(f"Adjacency: {len(adj_df)} edges")
    return df, adj_df


# ---------------------------------------------------------------------------
# Feature engineering (S -> R transfer)
# ---------------------------------------------------------------------------

def compute_spatial_lags(
    df: pd.DataFrame,
    acs_cols: list,
    adj_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean-neighbor ACS features."""
    id_col = "zcta_id" if "zcta_id" in df.columns else "zcta"

    # Build adjacency dict from edge-list dataframe
    zcta_col = "zcta" if "zcta" in adj_df.columns else "zcta_id"
    nbr_col = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"
    adjacency: dict = {}
    for row in adj_df.itertuples(index=False):
        z = str(getattr(row, zcta_col)).zfill(5)
        nb = str(getattr(row, nbr_col)).zfill(5)
        adjacency.setdefault(z, []).append(nb)

    id_to_idx = {str(v).zfill(5): i for i, v in enumerate(df[id_col].values)}
    X = df[acs_cols].values.astype(np.float64)

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

    log.info(f"Spatial lag: {hit} with neighbors, {miss} without")
    lag_cols = {f"lag_{c}": lag_X[:, j] for j, c in enumerate(acs_cols)}
    return df.assign(**lag_cols)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discovered-structure features."""
    df = df.copy()
    pop = df["population"].values.astype(np.float64)
    df["struct_log_population"] = np.log1p(np.maximum(pop, 0))
    df["struct_latitude"] = df["latitude"].astype(np.float64)
    df["struct_longitude"] = df["longitude"].astype(np.float64)
    log.info("Added struct_log_population, struct_latitude, struct_longitude")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get all feature columns: acs_* + struct_* + lag_*."""
    skip = {
        "place", "county", "state", "latitude", "longitude", "Count_Person",
        "zcta", "county_fips", "state_fips", "imputation_split", "extrapolation_split",
        "superresolution_split", "split", "zcta_id", "county_name", "population",
        "split_imputation", "split_extrap", "split_superres",
        *[f"target_{t}" for t in SUPPORTED_TASK_NAMES],
    }
    cols = sorted(
        c for c in df.columns
        if (c.startswith("acs_") or c.startswith("struct_") or c.startswith("lag_"))
        and c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
    )
    n_acs = sum(1 for c in cols if c.startswith("acs_"))
    n_struct = sum(1 for c in cols if c.startswith("struct_"))
    n_lag = sum(1 for c in cols if c.startswith("lag_"))
    log.info(f"Feature columns: {len(cols)} ({n_acs} ACS + {n_struct} structure + {n_lag} lags)")
    return cols


def split_data(df: pd.DataFrame):
    """Split into train/test masks (same as v1)."""
    for col in ("split_imputation", "imputation_split", "split"):
        if col in df.columns:
            test_mask = (df[col] == "test").values
            train_mask = ~test_mask
            log.info(f"Split ({col}): train={train_mask.sum()}, test={test_mask.sum()}")
            return train_mask, test_mask
    rng = np.random.RandomState(42)
    train_mask = rng.rand(len(df)) < 0.8
    test_mask = ~train_mask
    log.info(f"Split (random): train={train_mask.sum()}, test={test_mask.sum()}")
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adj_norm(adj_df: pd.DataFrame, zcta_index: dict, n: int, device):
    """Build row-normalised sparse adjacency tensor."""
    zs = adj_df["zcta"].astype(str).str.zfill(5).values
    ns = adj_df["neighbor_zcta"].astype(str).str.zfill(5).values

    src_idx = np.array([zcta_index.get(z, -1) for z in zs], dtype=np.int64)
    tgt_idx = np.array([zcta_index.get(z, -1) for z in ns], dtype=np.int64)

    valid = (src_idx >= 0) & (tgt_idx >= 0)
    src_idx = src_idx[valid]
    tgt_idx = tgt_idx[valid]

    if len(src_idx) == 0:
        log.warning("No adjacency edges resolved -- using identity.")
        idx = torch.arange(n)
        vals = torch.ones(n)
        return torch.sparse_coo_tensor(
            torch.stack([idx, idx]), vals, (n, n)
        ).coalesce().to(device)

    rows_t = torch.from_numpy(src_idx)
    cols_t = torch.from_numpy(tgt_idx)
    vals_t = torch.ones(len(rows_t), dtype=torch.float32)

    A_cpu = torch.sparse_coo_tensor(
        torch.stack([rows_t, cols_t]), vals_t, (n, n)
    ).coalesce()
    deg = torch.sparse.sum(A_cpu, dim=1).to_dense().clamp(min=1.0)
    norm_vals = vals_t / deg[rows_t]

    adj_norm = torch.sparse_coo_tensor(
        torch.stack([rows_t, cols_t]).to(device),
        norm_vals.to(device),
        (n, n),
    ).coalesce()

    n_directed = int(valid.sum())
    log.info(f"Adjacency: {n} nodes, {n_directed // 2} undirected edges, "
             f"{n_directed / max(n, 1):.1f} avg degree")
    return adj_norm


# ---------------------------------------------------------------------------
# Stage 1: GNN training
# ---------------------------------------------------------------------------

def train_gnn_stage1(
    df: pd.DataFrame,
    adj_df: pd.DataFrame,
    feature_cols: list,
    train_mask: np.ndarray,
    device,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
):
    """Train GraphSAGE and return (model, latents, loss_history, feature_stats)."""
    n = len(df)
    id_col = "zcta_id" if "zcta_id" in df.columns else "zcta"
    zcta_index = {str(v).zfill(5): i for i, v in enumerate(df[id_col].values)}

    X_raw = df[feature_cols].values.astype(np.float32)
    medians = np.nanmedian(X_raw[train_mask], axis=0)
    for j in range(X_raw.shape[1]):
        nans = np.isnan(X_raw[:, j])
        if nans.any():
            X_raw[nans, j] = medians[j]

    mu = X_raw[train_mask].mean(axis=0)
    std = X_raw[train_mask].std(axis=0).clip(min=1e-8)
    X = (X_raw - mu) / std

    # Build target matrix
    task_names, Y_list = [], []
    for task in SUPPORTED_TASK_NAMES:
        col = f"target_{task}"
        if col not in df.columns:
            continue
        y = df[col].values.astype(np.float32)
        task_names.append(task)
        Y_list.append(y)

    Y = np.stack(Y_list, axis=1)
    Y_mu = np.nanmean(Y[train_mask], axis=0)
    Y_std = np.nanstd(Y[train_mask], axis=0).clip(min=1e-8)
    Y_norm = (Y - Y_mu) / Y_std

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y_norm).to(device)
    valid_t = torch.from_numpy(~np.isnan(Y)).to(device)
    train_t = torch.from_numpy(train_mask).to(device)

    adj_norm = build_adj_norm(adj_df, zcta_index, n, device)

    n_tasks = Y.shape[1]
    model = GNNModel(len(feature_cols), n_tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    loss_history = []

    log.info(f"Stage 1: GraphSAGE -- {n} nodes, {len(feature_cols)} features, "
             f"{n_tasks} tasks, {epochs} epochs, device={device}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        _, out = model(X_t, adj_norm)

        mask = train_t.unsqueeze(1) & valid_t
        Y_safe = torch.nan_to_num(Y_t, nan=0.0)
        diff = (out - Y_safe) ** 2
        loss = diff[mask].mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            log.info(f"  epoch {epoch}/{epochs}  loss={loss_val:.5f}  "
                     f"best={best_loss:.5f}  elapsed={elapsed:.0f}s")

        if no_improve >= patience:
            log.info(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    log.info(f"Stage 1 complete. Best loss={best_loss:.5f}  time={time.time() - t0:.0f}s")

    model.eval()
    with torch.no_grad():
        latents, _ = model(X_t, adj_norm)
    # Cast to float64 to match JSON tree walker precision (v1 fidelity lesson)
    latents_np = latents.cpu().numpy().astype(np.float64)

    feature_stats = {
        "means": mu.tolist(),
        "stds": std.tolist(),
        "medians": medians.tolist(),
    }

    return model, latents_np, loss_history, feature_stats


# ---------------------------------------------------------------------------
# Stage 2: GBDT + proxy certifier on frozen latents
# ---------------------------------------------------------------------------

def fit_proxy_certifier(z_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Fit Spearman-weighted proxy certifier."""
    n_dims = z_train.shape[1]
    weights = np.zeros(n_dims, dtype=np.float32)
    for j in range(n_dims):
        rho, _ = spearmanr(z_train[:, j], y_train)
        weights[j] = -rho if not np.isnan(rho) else 0.0
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights /= norm
    raw_scores = z_train @ weights
    kappas = 1.0 / (1.0 + np.exp(-raw_scores))
    edges = [float(np.quantile(kappas, q)) for q in [0.2, 0.4, 0.6, 0.8]]
    return {
        "feature_weights": weights.tolist(),
        "bucket_edges": edges,
        "sigma_scale": float(1.0 / max(np.std(z_train), 1e-6)),
    }


def train_stage2(
    df: pd.DataFrame,
    latents: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
):
    """Fit GBDT + proxy certifier per task on frozen GNN latents."""
    task_models = {}
    certifier_config = {}
    results = []

    log.info("Stage 2: GBDT + proxy certifier on 32D latents...")

    for task in SUPPORTED_TASK_NAMES:
        target_col = f"target_{task}"
        if target_col not in df.columns:
            continue

        y = df[target_col].values.astype(np.float64)
        valid = ~np.isnan(y)
        tr = train_mask & valid
        te = test_mask & valid

        if tr.sum() < 100:
            log.warning(f"  SKIP {task}: too few training samples ({tr.sum()})")
            continue

        z_tr, z_te = latents[tr], latents[te]
        y_tr, y_te = y[tr], y[te]

        gbdt = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
        )
        gbdt.fit(z_tr, y_tr)
        r2 = r2_score(y_te, gbdt.predict(z_te))

        task_models[task] = gbdt
        certifier_config[task] = fit_proxy_certifier(z_tr, y_tr)

        v1_r2 = V1_TASK_R2.get(task, 0.0)
        delta = r2 - v1_r2
        results.append({
            "task": task, "r2": round(r2, 4),
            "v1_r2": round(v1_r2, 4), "delta": round(delta, 4),
            "n_train": int(tr.sum()),
        })
        marker = "+" if delta > 0 else ""
        log.info(f"  {task}: R2={r2:.4f} (v1={v1_r2:.4f}, delta={marker}{delta:.4f})")

    log.info(f"Stage 2 complete. {len(task_models)} tasks.")
    v2_mean = np.mean([e["r2"] for e in results])
    v1_mean = np.mean([e["v1_r2"] for e in results])
    improved = sum(1 for e in results if e["delta"] > 0)
    log.info(f"  v2 mean R2: {v2_mean:.4f}  v1 mean: {v1_mean:.4f}  "
             f"delta: {v2_mean - v1_mean:+.4f}  improved: {improved}/{len(results)}")

    return task_models, certifier_config, results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_gnn_weights(model) -> dict:
    """Serialize GNN state_dict to JSON-compatible nested lists."""
    state = model.state_dict()
    gnn_weights = {}
    key_map = {
        "sage.encoder.0.weight": "encoder_weight",
        "sage.encoder.0.bias": "encoder_bias",
        "sage.encoder.1.weight": "encoder_ln_weight",
        "sage.encoder.1.bias": "encoder_ln_bias",
        "sage.sage1.linear.weight": "conv1_weight",
        "sage.sage1.linear.bias": "conv1_bias",
        "sage.norm1.weight": "norm1_weight",
        "sage.norm1.bias": "norm1_bias",
        "sage.sage2.linear.weight": "conv2_weight",
        "sage.sage2.linear.bias": "conv2_bias",
    }
    for torch_key, json_key in key_map.items():
        if torch_key in state:
            tensor = state[torch_key].cpu().numpy()
            gnn_weights[json_key] = tensor.tolist()
    return gnn_weights


def _extract_tree(estimator) -> dict:
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
    if "v" in node:
        return node["v"]
    if x[node["f"]] <= node["t"]:
        return _walk(node["l"], x)
    return _walk(node["r"], x)


def predict_json(gbdt_json: dict, X: np.ndarray) -> np.ndarray:
    preds = np.full(len(X), gbdt_json["init"])
    lr = gbdt_json["lr"]
    for tree in gbdt_json["trees"]:
        for i in range(len(X)):
            preds[i] += lr * _walk(tree, X[i])
    return preds


def validate_fidelity(task_models: dict, serialized: dict, z: np.ndarray,
                      n_samples: int = 500, tolerance: float = 1e-6) -> dict:
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

def build_artifact(
    gnn_weights: dict,
    feature_cols: list,
    feature_stats: dict,
    task_models: dict,
    certifier_config: dict,
    task_results: list,
    latents: np.ndarray,
    loss_history: list,
    feature_breakdown: dict,
) -> dict:
    log.info("Serializing GBDTs...")
    predictors = {}
    for task, model in task_models.items():
        predictors[task] = serialize_gbdt(model)
    log.info(f"  Serialized {len(predictors)} task GBDTs")

    fidelity = validate_fidelity(task_models, predictors, latents)

    task_metrics = {}
    for entry in task_results:
        task_metrics[entry["task"]] = {
            "r2": entry["r2"], "v1_r2": entry["v1_r2"],
            "delta": entry["delta"], "n_train": entry["n_train"],
        }

    mean_r2 = np.mean([e["r2"] for e in task_results]) if task_results else 0.0
    v1_mean = np.mean([e["v1_r2"] for e in task_results]) if task_results else 0.0
    fb = feature_breakdown

    return {
        "format": "geo_cert_gnn_v2",
        "version": VERSION,
        "model_key": MODEL_KEY,
        "created": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "parent_model": "geo_cert_gnn_v1",
            "parent_version": "1.0.0",
            "change_type": "feature_expansion",
            "change_summary": (
                f"Expanded GNN input from {fb.get('n_acs', 33)} ACS to "
                f"{fb['n_acs']} ACS + {fb['n_struct']} structure + {fb['n_lag']} lag features. "
                f"Same architecture, same GBDT hyperparams. S->R transfer."
            ),
        },
        "representation": {
            "type": "graphsage_2layer_v2_structure",
            "feature_order": feature_cols,
            "n_features": len(feature_cols),
            "n_components": 32,
            "hidden_dim": 64,
            "feature_means": feature_stats["means"],
            "feature_stds": feature_stats["stds"],
            "feature_medians": feature_stats["medians"],
            "gnn_weights": gnn_weights,
        },
        "predictors": predictors,
        "certifier": certifier_config,
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
            "inference_deps": "torch + numpy",
            "sklearn_version": str(__import__("sklearn").__version__),
            "torch_version": torch.__version__ if HAS_TORCH else "N/A",
            "training": {
                "epochs_run": len(loss_history),
                "final_loss": loss_history[-1] if loss_history else None,
                "best_loss": min(loss_history) if loss_history else None,
            },
        },
    }


def build_manifest(artifact_bytes: bytes) -> dict:
    sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "generated_by": "apps/geo_cert/train_and_export_gnn_v2.py",
        "generated_at": now,
        "artifact_id": MODEL_KEY,
        "engine_id": ENGINE_ID,
        "encoder_id": "graphsage_2layer_structure_32d",
        "task_type": "geospatial",
        "version": VERSION,
        "artifact_uri": f"s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}",
        "sha256": sha256,
        "file_format": "json",
        "src_dim": 65,
        "projection_dim": 32,
        "serving_path": "geo_cert_v1",
        "validation_status": "experimental",
        "training_run_id": os.environ.get("TRAINING_JOB_NAME"),
        "created_at": now,
        "classifier_source": "geo_gbdt_proxy",
        "canonical_name": "geo_cert_gnn_structure_v2",
        "metadata": {
            "task_family": "geospatial",
            "compatible_endpoints": ["/geo/score"],
            "decomposition_mode": "proxy_spearman_v2",
            "encoder_architecture": "graphsage_2layer_mean_agg",
            "lineage_parent": "geo_cert_gnn_v1",
        },
    }


def validate_manifest(manifest: dict) -> bool:
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
# Timestamped results artifact
# ---------------------------------------------------------------------------

def build_results_artifact(task_results: list, artifact_sha256: str,
                           feature_breakdown: dict, loss_history: list) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "run_id": f"gnn_v2_{now.strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": now.isoformat(),
        "model_version": VERSION,
        "engine_id": ENGINE_ID,
        "parent_model": "geo_cert_gnn_v1",
        "artifact_sha256": artifact_sha256,
        "feature_set": {
            "n_acs": feature_breakdown["n_acs"],
            "n_structure": feature_breakdown["n_struct"],
            "n_spatial_lag": feature_breakdown["n_lag"],
            "total": sum(feature_breakdown.values()),
            "gnn_latent_dim": 32,
            "gnn_hidden_dim": 64,
        },
        "hyperparameters": {
            "gnn_epochs_max": 300,
            "gnn_lr": 1e-3,
            "gnn_weight_decay": 1e-4,
            "gnn_patience": 30,
            "gbdt_n_estimators": 200,
            "gbdt_max_depth": 4,
            "gbdt_learning_rate": 0.05,
            "gbdt_min_samples_leaf": 40,
        },
        "training": {
            "epochs_run": len(loss_history),
            "best_loss": min(loss_history) if loss_history else None,
        },
        "split": "split_imputation (geographic)",
        "per_task_results": task_results,
        "aggregate": {
            "v2_mean_r2": round(float(np.mean([e["r2"] for e in task_results])), 4),
            "v1_mean_r2": round(float(np.mean([e["v1_r2"] for e in task_results])), 4),
            "mean_delta": round(float(np.mean([e["delta"] for e in task_results])), 4),
            "tasks_improved": sum(1 for e in task_results if e["delta"] > 0),
            "tasks_regressed": sum(1 for e in task_results if e["delta"] < 0),
            "total_tasks": len(task_results),
        },
    }


# ---------------------------------------------------------------------------
# Upload + registry
# ---------------------------------------------------------------------------

def upload(artifact_bytes: bytes, manifest: dict, results: dict) -> None:
    s3 = boto3.client("s3", region_name=REGION)

    log.info(f"Uploading artifact ({len(artifact_bytes)/1024:.1f} KB) "
             f"to s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=OUTPUT_KEY, Body=artifact_bytes,
                  ContentType="application/json")

    manifest_bytes = json.dumps(manifest, indent=2).encode()
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=MANIFEST_KEY, Body=manifest_bytes,
                  ContentType="application/json")

    results_key = f"geo-cert/runs/{results['run_id']}.results.json"
    results_bytes = json.dumps(results, indent=2).encode()
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=results_key, Body=results_bytes,
                  ContentType="application/json")
    log.info(f"Uploaded results to s3://{OUTPUT_BUCKET}/{results_key}")

    log.info("Verifying round-trip integrity...")
    resp = s3.get_object(Bucket=OUTPUT_BUCKET, Key=OUTPUT_KEY)
    downloaded = resp["Body"].read()
    actual_sha = hashlib.sha256(downloaded).hexdigest()
    if actual_sha != manifest["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: expected {manifest['sha256'][:16]}, got {actual_sha[:16]}")
    log.info(f"  SHA256 verified: {actual_sha[:16]}...")


def build_engine_entry(manifest: dict) -> dict:
    return {
        "provider": "geo_cert",
        "model_name": "graphsage_structure_gbdt_v2",
        "src_dim": manifest["src_dim"],
        "artifact_uri": manifest["artifact_uri"].replace(f"s3://{OUTPUT_BUCKET}/", ""),
        "onnx_uri": "",
        "artifact_layout": "bundled_v2",
        "encoder_family": "graphsage_2layer_structure_32d",
        "runtime_profile": "python_local",
        "preferred_artifact_format": "json",
        "allowed_artifact_formats": ["json"],
        "validation_status": "experimental",
        "status": "active",
        "task_family": "geospatial",
        "compatible_endpoints": ["/geo/score"],
        "decomposition_mode": "proxy_spearman_v2",
        "input_contract": "zcta_features_v2_adjacency",
        "supports_runtime_embedding": False,
        "served_publicly": False,
        "requires_pair_input": False,
        "supports_text_input": False,
        "supports_embedding_input": False,
        "tier_allowlist": [],
        "deprecated": False,
        "metadata": {
            "classifier_source": "geo_gbdt_proxy",
            "serving_path": "geo_cert_v1",
            "sha256": manifest["sha256"],
            "encoder_architecture": "graphsage_2layer_mean_agg",
            "projection_dim": 32,
            "benchmark": "CONUS-27",
            "lineage_parent": "geo_cert_gnn_v1",
        },
    }


def update_engine_registry(manifest: dict, dry_run: bool = False) -> None:
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
        log.info(f"  ModelRegistry validation: PASS (engine_id={spec.engine_id})")
    except ImportError:
        log.warning("  yrsn not installed -- skipping ModelRegistry validation")
    except Exception as e:
        log.error(f"  ModelRegistry validation FAILED: {e}")
        raise

    if dry_run:
        log.info(f"  [DRY RUN] Would update s3://{OUTPUT_BUCKET}/{ENGINES_KEY}")
        return

    body = json.dumps(registry, indent=2, default=str)
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=ENGINES_KEY, Body=body.encode(),
                  ContentType="application/json")
    log.info(f"  Updated s3://{OUTPUT_BUCKET}/{ENGINES_KEY}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HAS_TORCH:
        raise ImportError("torch is required. Install with: pip install torch")

    parser = argparse.ArgumentParser(
        description="geo_cert GNN v2: GraphSAGE + discovered structure (S->R transfer)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-local", type=str)
    parser.add_argument("--skip-registry", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--local-features", type=str)
    parser.add_argument("--local-adjacency", type=str)
    parser.add_argument("--compare-v1", action="store_true")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("geo_cert GNN v2 -- Train + Export (S -> R Transfer)")
    log.info("=" * 65)
    log.info("Feature expansion: ACS + structure + spatial lags -> GraphSAGE -> 32D")
    log.info("Same architecture, same hyperparams as v1 -- isolating feature effect")
    log.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 1. Load data
    df, adj_df = load_data(args.local_features, args.local_adjacency)

    # 2. Compute spatial lags
    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
    )
    df = compute_spatial_lags(df, acs_cols, adj_df)

    # 3. Add structure features
    df = add_structure_features(df)

    # 3b. Spatial representation QA (Priority 2 from representation_qa_principle.md)
    id_col = "zcta_id" if "zcta_id" in df.columns else "zcta"
    _zcta_c = "zcta" if "zcta" in adj_df.columns else "zcta_id"
    _nbr_c = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"
    _adj_dict = {}
    for _row in adj_df.itertuples(index=False):
        _z = str(getattr(_row, _zcta_c)).zfill(5)
        _nb = str(getattr(_row, _nbr_c)).zfill(5)
        _adj_dict.setdefault(_z, []).append(_nb)
    run_spatial_validation(df, acs_cols, _adj_dict, output_dir=args.output_local, id_col=id_col)

    # 4. Get expanded feature columns
    feature_cols = get_feature_cols(df)
    train_mask, test_mask = split_data(df)

    n_acs = sum(1 for c in feature_cols if c.startswith("acs_"))
    n_struct = sum(1 for c in feature_cols if c.startswith("struct_"))
    n_lag = sum(1 for c in feature_cols if c.startswith("lag_"))
    feature_breakdown = {"n_acs": n_acs, "n_struct": n_struct, "n_lag": n_lag}

    # 5. Stage 1: GNN training
    model, latents, loss_history, feature_stats = train_gnn_stage1(
        df, adj_df, feature_cols, train_mask, device, epochs=args.epochs,
    )

    # 6. Stage 2: GBDT on frozen latents
    task_models, certifier_config, task_results = train_stage2(
        df, latents, train_mask, test_mask,
    )

    # 6b. Save torch state_dict for future inference (load_state_dict path)
    if args.output_local:
        pt_path = Path(args.output_local) / "gnn_sage_v2.pt"
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.sage.state_dict(), str(pt_path))
        log.info(f"Saved GNN state_dict to {pt_path}")

    # 7. Serialize
    gnn_weights = serialize_gnn_weights(model)
    artifact = build_artifact(
        gnn_weights=gnn_weights,
        feature_cols=feature_cols,
        feature_stats=feature_stats,
        task_models=task_models,
        certifier_config=certifier_config,
        task_results=task_results,
        latents=latents,
        loss_history=loss_history,
        feature_breakdown=feature_breakdown,
    )
    artifact_bytes = json.dumps(artifact).encode()
    manifest = build_manifest(artifact_bytes)
    results = build_results_artifact(task_results, manifest["sha256"],
                                     feature_breakdown, loss_history)

    log.info(f"\n{'=' * 65}")
    log.info(f"ARTIFACT SUMMARY")
    log.info(f"{'=' * 65}")
    log.info(f"Artifact:   {len(artifact_bytes)/1024:.1f} KB")
    log.info(f"SHA256:     {manifest['sha256'][:16]}...")
    log.info(f"Tasks:      {artifact['metadata']['n_tasks_trained']}")
    log.info(f"v2 mean R2: {artifact['metadata']['mean_r2']}")
    log.info(f"v1 mean R2: {artifact['metadata']['v1_mean_r2']}")
    log.info(f"Delta:      {artifact['metadata']['mean_r2_delta']:+.4f}")
    log.info(f"GNN:        {len(feature_cols)} features -> 64 -> 32D latents")
    log.info(f"Epochs:     {len(loss_history)}")

    valid = validate_manifest(manifest)
    if not valid:
        log.error("Manifest validation failed")
        return

    if args.compare_v1:
        print(f"\n{'=' * 65}")
        print(f"PER-TASK GNN v1 vs v2 COMPARISON")
        print(f"{'=' * 65}")
        print(f"{'Task':<30} {'v1 R2':>8} {'v2 R2':>8} {'Delta':>8} {'Result':>8}")
        print(f"{'-' * 65}")
        for entry in sorted(task_results, key=lambda e: e["delta"], reverse=True):
            marker = "[+]" if entry["delta"] > 0.005 else ("[-]" if entry["delta"] < -0.005 else "[=]")
            print(f"{entry['task']:<30} {entry['v1_r2']:>8.4f} {entry['r2']:>8.4f} "
                  f"{entry['delta']:>+8.4f} {marker:>8}")
        agg = results["aggregate"]
        print(f"{'-' * 65}")
        print(f"{'MEAN':<30} {agg['v1_mean_r2']:>8.4f} {agg['v2_mean_r2']:>8.4f} "
              f"{agg['mean_delta']:>+8.4f}")
        print(f"Improved: {agg['tasks_improved']}/{agg['total_tasks']}  "
              f"Regressed: {agg['tasks_regressed']}/{agg['total_tasks']}")

    if args.output_local:
        out = Path(args.output_local)
        out.mkdir(parents=True, exist_ok=True)
        (out / "geo_cert_gnn_v2.json").write_bytes(artifact_bytes)
        (out / "manifest_gnn_v2.json").write_text(json.dumps(manifest, indent=2))
        (out / f"{results['run_id']}.results.json").write_text(json.dumps(results, indent=2))
        log.info(f"\nWritten to {out}/")
        return

    if args.dry_run:
        log.info("\n[DRY RUN] Would upload to:")
        log.info(f"  s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
        log.info(f"  s3://{OUTPUT_BUCKET}/{MANIFEST_KEY}")
        log.info(f"  s3://{OUTPUT_BUCKET}/geo-cert/runs/{results['run_id']}.results.json")
        if not args.skip_registry:
            update_engine_registry(manifest, dry_run=True)
        return

    upload(artifact_bytes, manifest, results)

    # Upload .pt state_dict alongside JSON artifact
    import io
    pt_buffer = io.BytesIO()
    torch.save(model.sage.state_dict(), pt_buffer)
    pt_bytes = pt_buffer.getvalue()
    pt_key = OUTPUT_KEY.replace(".json", ".pt")
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=OUTPUT_BUCKET, Key=pt_key, Body=pt_bytes,
                  ContentType="application/octet-stream")
    log.info(f"Uploaded state_dict to s3://{OUTPUT_BUCKET}/{pt_key}")

    if not args.skip_registry:
        update_engine_registry(manifest, dry_run=False)

    log.info(f"\n{'=' * 65}")
    log.info("EXPORT COMPLETE")
    log.info(f"  Artifact:  s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
    log.info(f"  StateDict: s3://{OUTPUT_BUCKET}/{pt_key}")
    log.info(f"  Manifest:  s3://{OUTPUT_BUCKET}/{MANIFEST_KEY}")
    log.info(f"  Results:   s3://{OUTPUT_BUCKET}/geo-cert/runs/{results['run_id']}.results.json")
    log.info(f"  Engine:    {manifest['engine_id']}")
    log.info(f"  SHA256:    {manifest['sha256'][:16]}...")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
