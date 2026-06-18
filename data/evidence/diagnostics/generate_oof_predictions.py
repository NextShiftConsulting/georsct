#!/usr/bin/env python3
"""
generate_oof_predictions.py — Generate OOF prediction parquets for TRF estimation.

Retrains each ceiling model architecture on the standard split, captures
test-set predictions, and writes one parquet per model in ceiling_schema format.

Models generated:
  - oof_pca_v1.parquet     — PCA-32 on 31 ACS features (v1 baseline)
  - oof_pca_v2.parquet     — PCA-32 on 65 features (ACS + structure + lags)
  - oof_gnn_v1.parquet     — GraphSAGE on 33 ACS features → 32D → GBDT
  - oof_gnn_v2.parquet     — GraphSAGE on 69 features → 32D → GBDT
  - oof_spatial_lag_v1.parquet — Spatial lag model (PCA on ACS + lags)

Why retrain instead of loading artifacts?
  GNN artifacts store serialized GBDT trees on 32D latents, but reproducing
  the GNN forward pass from JSON weights requires torch + graph construction.
  Retraining is fast (~2 min per PCA model, ~1 min GNN on CPU) and guarantees
  the OOF predictions match the exact train/test split.

Usage:
    python generate_oof_predictions.py --local-data C:/tmp/geo_data --output-dir C:/tmp/oof_predictions
    python generate_oof_predictions.py --models pca_v1,pca_v2  # subset
"""

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler

from apps.geo_cert.models.ceiling.ceiling_schema import (
    VALID_MODEL_VERSIONS,
    build_oof_rows,
    validate,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

REGION = "us-east-1"
DATA_BUCKET = "yrsn-datasets"
DATA_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
ADJACENCY_KEY = "rsct_curriculum/series_018/processed/zcta_adjacency.parquet"

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

# State abbreviation to FIPS mapping (48 contiguous + DC)
STATE_ABBREV_TO_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20",
    "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25",
    "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
    "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51",
    "WA": "53", "WV": "54", "WI": "55", "WY": "56",
    # Territories that may appear
    "AK": "02", "HI": "15", "PR": "72", "VI": "78", "GU": "66",
    "AS": "60", "MP": "69",
}


# ---------------------------------------------------------------------------
# Data loading (shared across all models)
# ---------------------------------------------------------------------------

def load_data(local_path: str = None) -> pd.DataFrame:
    """Load ZCTA features+labels."""
    if local_path:
        p = Path(local_path)
        if p.is_dir():
            candidate = p / "zcta_features_labels.parquet"
            if candidate.exists():
                log.info(f"Loading features from {candidate}")
                return pd.read_parquet(candidate)
        elif p.exists():
            return pd.read_parquet(p)
    s3 = boto3.client("s3", region_name=REGION)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    s3.download_file(DATA_BUCKET, DATA_KEY, tmp.name)
    tmp.close()
    df = pd.read_parquet(tmp.name)
    os.unlink(tmp.name)
    return df


def load_adjacency(local_path: str = None) -> pd.DataFrame:
    """Load adjacency edge list as DataFrame."""
    if local_path:
        p = Path(local_path)
        if p.is_dir():
            candidate = p / "zcta_adjacency.parquet"
            if candidate.exists():
                return pd.read_parquet(candidate)
        elif p.exists():
            return pd.read_parquet(p)
    s3 = boto3.client("s3", region_name=REGION)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    s3.download_file(DATA_BUCKET, ADJACENCY_KEY, tmp.name)
    tmp.close()
    df = pd.read_parquet(tmp.name)
    os.unlink(tmp.name)
    return df


def get_split_masks(df: pd.DataFrame):
    """Return train_mask, test_mask arrays."""
    test_mask = (df["split_imputation"] == "test").values
    train_mask = ~test_mask
    return train_mask, test_mask


def get_state_fips(df: pd.DataFrame) -> np.ndarray:
    """Convert state abbreviations to 2-digit FIPS codes."""
    return np.array([
        STATE_ABBREV_TO_FIPS.get(s, "99") for s in df["state"].values
    ])


def get_leakage_safe_cols(feature_cols: list, task: str) -> list:
    """Remove leakage-prone features."""
    if task in HEALTH_TASKS:
        return [c for c in feature_cols
                if c != "acs_pct_no_insurance"
                and c != "lag_acs_pct_no_insurance"]
    if task in ("home_value", "income"):
        return [c for c in feature_cols
                if "income" not in c.lower() and "poverty" not in c.lower()]
    return list(feature_cols)


# ---------------------------------------------------------------------------
# Feature engineering
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
    lag_cols = {f"lag_{c}": lag_X[:, j] for j, c in enumerate(feature_cols)}
    return df.assign(**lag_cols)


def adjacency_df_to_dict(adj_df: pd.DataFrame) -> dict:
    """Convert adjacency DataFrame to {zcta: [neighbors]}."""
    zcta_col = "zcta" if "zcta" in adj_df.columns else "zcta_id"
    nbr_col = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"
    result = {}
    for row in adj_df.itertuples(index=False):
        z = str(getattr(row, zcta_col)).zfill(5)
        nb = str(getattr(row, nbr_col)).zfill(5)
        result.setdefault(z, []).append(nb)
    return result


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add structure features: log_population, latitude, longitude."""
    df = df.copy()
    pop = df["population"].values.astype(np.float64)
    df["struct_log_population"] = np.log1p(np.maximum(pop, 0))
    df["struct_latitude"] = df["latitude"].astype(np.float64)
    df["struct_longitude"] = df["longitude"].astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Model-specific OOF generators
# ---------------------------------------------------------------------------

def _pca_gbdt_oof(
    df: pd.DataFrame,
    feature_cols: list,
    model_version: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    zcta_ids: np.ndarray,
    state_fips: np.ndarray,
) -> pd.DataFrame:
    """Train PCA-32 + GBDT, return test-set OOF predictions as ceiling_schema DataFrame."""
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
    z_all = pipe.transform(X)
    log.info(f"  PCA: {X.shape[1]} features -> {n_components} components")

    all_rows = []
    for task in SUPPORTED_TASK_NAMES:
        target_col = f"target_{task}"
        if target_col not in df.columns:
            continue
        y = df[target_col].values.astype(np.float64)
        valid = ~np.isnan(y)
        tr = train_mask & valid
        te = test_mask & valid
        if tr.sum() < 100:
            continue

        gbdt = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
        )
        gbdt.fit(z_all[tr], y[tr])
        y_pred = gbdt.predict(z_all[te])
        y_true = y[te]
        r2 = r2_score(y_true, y_pred)

        rows = build_oof_rows(
            zctas=zcta_ids[te],
            task=task,
            fold="test",
            y_true=y_true,
            y_pred=y_pred,
            model_version=model_version,
            state_fips=state_fips[te],
        )
        all_rows.append(rows)
        log.info(f"    {task}: R²={r2:.4f} ({te.sum()} test samples)")

    return pd.concat(all_rows, ignore_index=True)


def _gnn_gbdt_oof(
    df: pd.DataFrame,
    feature_cols: list,
    adj_df: pd.DataFrame,
    model_version: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    zcta_ids: np.ndarray,
    state_fips: np.ndarray,
    epochs: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Train GraphSAGE + GBDT, return test-set OOF predictions.

    Architecture matches train_and_export_gnn.py / train_and_export_gnn_v2.py exactly:
      - SAGEConv: cat([x, neigh]) → Linear(in_dim*2, out_dim)
      - Row-normalized adjacency (D^{-1}A), no self-loops
      - GraphSAGE: encoder(n_feat→64) → SAGEConv(64→64) → SAGEConv(64→32)
      - Per-task linear heads (ModuleList)

    Args:
        seed: Random seed for GNN initialization. Different seeds produce
              different latent spaces → different GBDT predictions. Use
              multiple seeds and average R² at estimator time.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    n = len(df)

    zcta_index = {str(v).zfill(5): i for i, v in enumerate(df["zcta_id"].values)}

    # Normalize features (same as train scripts)
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
        task_names.append(task)
        Y_list.append(df[col].values.astype(np.float32))
    Y = np.stack(Y_list, axis=1)
    Y_mu = np.nanmean(Y[train_mask], axis=0)
    Y_std = np.nanstd(Y[train_mask], axis=0).clip(min=1e-8)
    Y_norm = (Y - Y_mu) / Y_std

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y_norm).to(device)
    valid_t = torch.from_numpy(~np.isnan(Y)).to(device)
    train_t = torch.from_numpy(train_mask).to(device)

    # Build row-normalized adjacency (D^{-1}A) — matches train scripts exactly
    zcta_col = "zcta" if "zcta" in adj_df.columns else "zcta_id"
    nbr_col = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"
    zs = adj_df[zcta_col].astype(str).str.zfill(5).values
    ns = adj_df[nbr_col].astype(str).str.zfill(5).values
    src_idx = np.array([zcta_index.get(z, -1) for z in zs], dtype=np.int64)
    tgt_idx = np.array([zcta_index.get(z, -1) for z in ns], dtype=np.int64)
    valid_edges = (src_idx >= 0) & (tgt_idx >= 0)
    src_idx = src_idx[valid_edges]
    tgt_idx = tgt_idx[valid_edges]

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

    n_directed = int(valid_edges.sum())
    log.info(f"  Adjacency: {n} nodes, {n_directed // 2} undirected edges, "
             f"{n_directed / max(n, 1):.1f} avg degree")

    # GNN architecture — exact match to train_and_export_gnn*.py
    class SAGEConv(nn.Module):
        def __init__(self, in_dim, out_dim, bias=True):
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim, bias=bias)

        def forward(self, x, adj):
            neigh = torch.sparse.mm(adj, x)
            agg = torch.cat([x, neigh], dim=-1)
            return self.linear(agg)

    class GraphSAGE(nn.Module):
        def __init__(self, n_feat, hidden=64, out_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            self.sage1 = SAGEConv(hidden, hidden)
            self.sage2 = SAGEConv(hidden, out_dim)
            self.norm1 = nn.LayerNorm(hidden)

        def forward(self, x, adj):
            h = self.encoder(x)
            h = F.relu(self.norm1(self.sage1(h, adj)))
            h = self.sage2(h, adj)
            return h

    class GNNModel(nn.Module):
        def __init__(self, n_feat, n_tasks, hidden=64, latent=32):
            super().__init__()
            self.sage = GraphSAGE(n_feat, hidden, latent)
            self.heads = nn.ModuleList([nn.Linear(latent, 1) for _ in range(n_tasks)])

        def forward(self, x, adj):
            z = self.sage(x, adj)
            out = torch.cat([h(z) for h in self.heads], dim=-1)
            return z, out

    n_tasks = Y.shape[1]
    model = GNNModel(len(feature_cols), n_tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_state = None
    patience = 50
    no_improve = 0

    log.info(f"  GNN training: {n} nodes, {len(feature_cols)} features, "
             f"{n_tasks} tasks, {epochs} epochs")
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

        lv = loss.item()
        if lv < best_loss - 1e-5:
            best_loss = lv
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            log.info(f"    epoch {epoch}/{epochs}  loss={lv:.5f}  "
                     f"best={best_loss:.5f}  elapsed={time.time()-t0:.0f}s")

        if no_improve >= patience:
            log.info(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        latents, _ = model(X_t, adj_norm)
    # float64 to match JSON tree walker precision (fidelity lesson from v1)
    latents_np = latents.cpu().numpy().astype(np.float64)

    # Stage 2: GBDT on frozen latents
    all_rows = []
    for ti, task in enumerate(task_names):
        target_col = f"target_{task}"
        y = df[target_col].values.astype(np.float64)
        valid = ~np.isnan(y)
        tr = train_mask & valid
        te = test_mask & valid
        if tr.sum() < 100:
            continue

        gbdt = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
        )
        gbdt.fit(latents_np[tr], y[tr])
        y_pred = gbdt.predict(latents_np[te])
        y_true = y[te]
        r2 = r2_score(y_true, y_pred)

        rows = build_oof_rows(
            zctas=zcta_ids[te],
            task=task,
            fold="test",
            y_true=y_true,
            y_pred=y_pred,
            model_version=model_version,
            state_fips=state_fips[te],
        )
        all_rows.append(rows)
        log.info(f"    {task}: R²={r2:.4f} ({te.sum()} test samples)")

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Spatial lag model (PCA on ACS + lag features, no structure features)
# ---------------------------------------------------------------------------

def _spatial_lag_oof(
    df: pd.DataFrame,
    acs_cols: list,
    adjacency: dict,
    model_version: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    zcta_ids: np.ndarray,
    state_fips: np.ndarray,
) -> pd.DataFrame:
    """Spatial lag v1: PCA on ACS + lag features (no struct features)."""
    df = compute_spatial_lags(df, acs_cols, adjacency)
    lag_cols = sorted(c for c in df.columns if c.startswith("lag_acs_"))
    feature_cols = acs_cols + lag_cols
    log.info(f"  Spatial lag features: {len(acs_cols)} ACS + {len(lag_cols)} lags = {len(feature_cols)}")
    return _pca_gbdt_oof(
        df, feature_cols, model_version,
        train_mask, test_mask, zcta_ids, state_fips,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

ALL_MODELS = {"pca_v1", "pca_v2", "spatial_lag_v1", "gnn_v1", "gnn_v2"}


GNN_DEFAULT_SEEDS = [0, 1, 2]


def generate_all(
    local_data: str = None,
    output_dir: str = "C:/tmp/oof_predictions",
    models: set = None,
    gnn_seeds: list = None,
):
    """Generate OOF predictions for specified models.

    Args:
        gnn_seeds: List of seeds for GNN models. Each seed produces a
            separate OOF parquet with model_version "gnn_v{X}_seed{N}".
            Default: [0, 1, 2] for 3-seed averaging at estimator time.
    """
    if models is None:
        models = ALL_MODELS
    if gnn_seeds is None:
        gnn_seeds = GNN_DEFAULT_SEEDS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load shared data
    df = load_data(local_data)
    train_mask, test_mask = get_split_masks(df)
    zcta_ids = df["zcta_id"].values
    state_fips = get_state_fips(df)

    log.info(f"Data: {len(df)} rows, train={train_mask.sum()}, test={test_mask.sum()}")
    log.info(f"States: {len(np.unique(state_fips))} unique FIPS codes")
    log.info(f"Models to generate: {sorted(models)}")

    # Identify ACS feature columns (shared by all models)
    skip = {
        "state", "latitude", "longitude", "zcta_id", "county_name",
        "population", "split_imputation", "split_extrap", "split_superres",
    }
    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
    )
    log.info(f"ACS feature columns: {len(acs_cols)}")

    # Load adjacency if needed by any model
    need_adjacency = models & {"pca_v2", "spatial_lag_v1", "gnn_v1", "gnn_v2"}
    adjacency_dict = None
    adj_df = None
    if need_adjacency:
        adj_df = load_adjacency(local_data)
        adjacency_dict = adjacency_df_to_dict(adj_df)
        log.info(f"Adjacency loaded: {len(adjacency_dict)} nodes")

    results = {}

    # --- PCA v1: 31 ACS features ---
    if "pca_v1" in models:
        log.info("\n=== PCA v1 ===")
        oof = _pca_gbdt_oof(
            df, acs_cols, "pca_v1",
            train_mask, test_mask, zcta_ids, state_fips,
        )
        path = out / "oof_pca_v1.parquet"
        oof.to_parquet(path, index=False)
        results["pca_v1"] = (path, len(oof))
        log.info(f"  Written {path}: {len(oof)} rows")

    # --- PCA v2: ACS + structure + lags ---
    if "pca_v2" in models:
        log.info("\n=== PCA v2 ===")
        df_v2 = add_structure_features(df)
        df_v2 = compute_spatial_lags(df_v2, acs_cols, adjacency_dict)
        lag_cols = sorted(c for c in df_v2.columns if c.startswith("lag_acs_"))
        struct_cols = sorted(c for c in df_v2.columns if c.startswith("struct_"))
        v2_feature_cols = acs_cols + struct_cols + lag_cols
        log.info(f"  v2 features: {len(acs_cols)} ACS + {len(struct_cols)} struct + {len(lag_cols)} lags = {len(v2_feature_cols)}")
        oof = _pca_gbdt_oof(
            df_v2, v2_feature_cols, "pca_v2",
            train_mask, test_mask, zcta_ids, state_fips,
        )
        path = out / "oof_pca_v2.parquet"
        oof.to_parquet(path, index=False)
        results["pca_v2"] = (path, len(oof))
        log.info(f"  Written {path}: {len(oof)} rows")

    # --- Spatial lag v1 ---
    if "spatial_lag_v1" in models:
        log.info("\n=== Spatial Lag v1 ===")
        oof = _spatial_lag_oof(
            df.copy(), acs_cols, adjacency_dict, "spatial_lag_v1",
            train_mask, test_mask, zcta_ids, state_fips,
        )
        path = out / "oof_spatial_lag_v1.parquet"
        oof.to_parquet(path, index=False)
        results["spatial_lag_v1"] = (path, len(oof))
        log.info(f"  Written {path}: {len(oof)} rows")

    # --- GNN v1: 33 ACS features (multi-seed) ---
    if "gnn_v1" in models:
        for seed in gnn_seeds:
            version = f"gnn_v1_seed{seed}"
            log.info(f"\n=== GNN v1 seed={seed} ===")
            oof = _gnn_gbdt_oof(
                df, acs_cols, adj_df, version,
                train_mask, test_mask, zcta_ids, state_fips,
                seed=seed,
            )
            path = out / f"oof_gnn_v1_seed{seed}.parquet"
            oof.to_parquet(path, index=False)
            results[version] = (path, len(oof))
            log.info(f"  Written {path}: {len(oof)} rows")

    # --- GNN v2: ACS + structure + lags (multi-seed) ---
    if "gnn_v2" in models:
        df_v2 = add_structure_features(df)
        df_v2 = compute_spatial_lags(df_v2, acs_cols, adjacency_dict)
        lag_cols = sorted(c for c in df_v2.columns if c.startswith("lag_acs_"))
        struct_cols = sorted(c for c in df_v2.columns if c.startswith("struct_"))
        v2_feature_cols = acs_cols + struct_cols + lag_cols
        for seed in gnn_seeds:
            version = f"gnn_v2_seed{seed}"
            log.info(f"\n=== GNN v2 seed={seed} ===")
            oof = _gnn_gbdt_oof(
                df_v2, v2_feature_cols, adj_df, version,
                train_mask, test_mask, zcta_ids, state_fips,
                seed=seed,
            )
            path = out / f"oof_gnn_v2_seed{seed}.parquet"
            oof.to_parquet(path, index=False)
            results[version] = (path, len(oof))
            log.info(f"  Written {path}: {len(oof)} rows")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("OOF GENERATION SUMMARY")
    log.info("=" * 60)
    for mv, (path, n_rows) in sorted(results.items()):
        log.info(f"  {mv}: {n_rows} rows -> {path}")
    log.info(f"Total: {sum(n for _, n in results.values())} rows across {len(results)} models")

    # Validate all outputs
    for mv, (path, _) in results.items():
        oof = pd.read_parquet(path)
        errors = validate(oof, strict=False)
        if errors:
            log.error(f"  VALIDATION FAIL {mv}: {errors}")
        else:
            log.info(f"  VALIDATION PASS {mv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOF prediction parquets for TRF estimation."
    )
    parser.add_argument(
        "--local-data", type=str, default="C:/tmp/geo_data",
        help="Local data directory (default: C:/tmp/geo_data)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="C:/tmp/oof_predictions",
        help="Output directory for parquets (default: C:/tmp/oof_predictions)"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model list (default: all). "
             "Options: pca_v1,pca_v2,spatial_lag_v1,gnn_v1,gnn_v2"
    )
    parser.add_argument(
        "--gnn-seeds", type=str, default="0,1,2",
        help="Comma-separated seeds for GNN models (default: 0,1,2). "
             "Each seed produces a separate OOF parquet."
    )
    args = parser.parse_args()

    models = None
    if args.models:
        models = set(args.models.split(","))
        invalid = models - ALL_MODELS
        if invalid:
            parser.error(f"Unknown models: {invalid}. Valid: {sorted(ALL_MODELS)}")

    gnn_seeds = [int(s) for s in args.gnn_seeds.split(",")]

    generate_all(
        local_data=args.local_data,
        output_dir=args.output_dir,
        models=models,
        gnn_seeds=gnn_seeds,
    )


if __name__ == "__main__":
    main()
