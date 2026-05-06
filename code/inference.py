#!/usr/bin/env python3
"""
geo_cert inference — Score ZCTAs using trained geo_cert models.

PCA models: pure numpy inference from JSON artifacts (no ML deps).
GNN models: reconstruct GraphSAGE from JSON weights (requires torch).

Usage:
    from apps.geo_cert.inference import GeoCertModel

    # PCA models — single ZCTA or batch, no setup needed
    model = GeoCertModel.from_s3("geo_cert_pca32_v1")
    pred = model.predict(zcta_features, "diabetes")
    batch = model.predict_batch(feature_matrix, "diabetes")

    # GNN models — must set_graph() before predict
    model = GeoCertModel.from_s3("geo_cert_gnn_v2")
    model.set_graph(X_all, adj_df, zcta_ids)
    batch = model.predict_batch(feature_matrix, "diabetes")

Models available:
    geo_cert_pca32_v1       — PCA-32 Champion (R²=0.655)
    geo_cert_spatial_lag_v1 — Spatial Lag Challenger (R²=0.666)
    geo_cert_gnn_v1         — GraphSAGE GNN Pilot (R²=0.651)
    geo_cert_gnn_v2         — GraphSAGE GNN v2 (R²=0.728)
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence

import boto3
import numpy as np

log = logging.getLogger(__name__)

BUCKET = "yrsn-checkpoints"
REGION = "us-east-1"

ARTIFACT_KEYS = {
    "geo_cert_pca32_v1": "geo-cert/geo_cert_pca32_v1.json",
    "geo_cert_spatial_lag_v1": "geo-cert/geo_cert_spatial_lag_v1.json",
    "geo_cert_gnn_v1": "geo-cert/geo_cert_gnn_v1.json",
    "geo_cert_pca32_v2": "geo-cert/geo_cert_pca32_v2.json",
    "geo_cert_gnn_v2": "geo-cert/geo_cert_gnn_v2.json",
}


def _walk(node: dict, x: np.ndarray) -> float:
    """Walk serialized GBDT tree node."""
    if "v" in node:
        return node["v"]
    if x[node["f"]] <= node["t"]:
        return _walk(node["l"], x)
    return _walk(node["r"], x)


class GeoCertModel:
    """Inference wrapper for geo_cert JSON artifacts (PCA and GNN)."""

    def __init__(self, artifact: dict, sha256: Optional[str] = None):
        self._artifact = artifact
        self._sha256 = sha256

        rep = artifact["representation"]
        self.feature_order = rep["feature_order"]
        self.n_features = rep["n_features"]
        self.n_components = rep["n_components"]
        self._tasks = list(artifact["predictors"].keys())

        # Detect artifact format: GNN has gnn_weights, PCA has components
        self._is_gnn = "gnn_weights" in rep

        if self._is_gnn:
            self._means = np.array(rep["feature_means"])
            self._stds = np.array(rep["feature_stds"])
            self._gnn_weights = rep["gnn_weights"]
            self._hidden_dim = rep.get("hidden_dim", 64)
            self._latent_cache = None  # Set by set_graph()
        else:
            self._means = np.array(rep["means"])
            self._stds = np.array(rep["stds"])
            self._components = np.array(rep["components"])  # (n_components, n_features)
            self._gnn_weights = None
            self._latent_cache = None

    @classmethod
    def from_s3(
        cls,
        model_name: str,
        bucket: str = BUCKET,
        region: str = REGION,
        verify_sha256: Optional[str] = None,
    ) -> "GeoCertModel":
        """Load model from S3.

        Args:
            model_name: One of geo_cert_pca32_v1, geo_cert_spatial_lag_v1, geo_cert_gnn_v1
            bucket: S3 bucket
            region: AWS region
            verify_sha256: If provided, verify artifact integrity
        """
        key = ARTIFACT_KEYS.get(model_name)
        if not key:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(ARTIFACT_KEYS.keys())}"
            )

        s3 = boto3.client("s3", region_name=region)
        log.info(f"Loading s3://{bucket}/{key}")
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()

        sha = hashlib.sha256(data).hexdigest()
        if verify_sha256 and sha != verify_sha256:
            raise RuntimeError(
                f"SHA256 mismatch: expected {verify_sha256[:16]}..., "
                f"got {sha[:16]}..."
            )

        artifact = json.loads(data)
        log.info(
            f"Loaded {model_name}: {len(artifact['predictors'])} tasks, "
            f"{artifact['representation']['n_components']}D"
        )
        return cls(artifact, sha256=sha)

    @classmethod
    def from_local(cls, path: str) -> "GeoCertModel":
        """Load model from local JSON file."""
        p = Path(path)
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        artifact = json.loads(data)
        return cls(artifact, sha256=sha)

    @property
    def is_gnn(self) -> bool:
        return self._is_gnn

    @property
    def tasks(self) -> list:
        """List of supported task names."""
        return list(self._tasks)

    @property
    def sha256(self) -> Optional[str]:
        return self._sha256

    @property
    def metadata(self) -> dict:
        return self._artifact.get("metadata", {})

    # ------------------------------------------------------------------
    # GNN graph setup
    # ------------------------------------------------------------------

    def set_graph(
        self,
        X_all: np.ndarray,
        adj_df,
        zcta_ids: Sequence[str],
    ) -> np.ndarray:
        """Precompute GNN latents for all nodes. Required before predict on GNN models.

        Args:
            X_all: (n_nodes, n_features) raw ACS features for ALL ZCTAs
            adj_df: DataFrame with (zcta, neighbor_zcta) edge list
            zcta_ids: ordered ZCTA IDs corresponding to rows of X_all
        Returns:
            (n_nodes, n_components) latent matrix
        """
        if not self._is_gnn:
            raise ValueError("set_graph() is only for GNN models; PCA models project on the fly")

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for GNN inference. pip install torch")

        # Normalize features
        X = np.asarray(X_all, dtype=np.float64)
        X = (X - self._means) / np.clip(self._stds, 1e-8, None)
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)

        n = len(X)
        zcta_index = {str(z).zfill(5): i for i, z in enumerate(zcta_ids)}
        adj_norm = self._build_adj_norm(adj_df, zcta_index, n)

        X_t = torch.from_numpy(X)
        with torch.no_grad():
            latents_t = self._gnn_forward(X_t, adj_norm)

        self._latent_cache = latents_t.numpy().astype(np.float64)
        log.info(f"GNN latents computed: {self._latent_cache.shape}")
        return self._latent_cache

    def _build_adj_norm(self, adj_df, zcta_index: dict, n: int):
        """Build row-normalized sparse adjacency tensor from edge-list DataFrame."""
        import torch

        zcta_col = "zcta" if "zcta" in adj_df.columns else "zcta_id"
        nbr_col = "neighbor_zcta" if "neighbor_zcta" in adj_df.columns else "neighbor"

        zs = adj_df[zcta_col].astype(str).str.zfill(5).values
        ns = adj_df[nbr_col].astype(str).str.zfill(5).values

        src_idx = np.array([zcta_index.get(z, -1) for z in zs], dtype=np.int64)
        tgt_idx = np.array([zcta_index.get(z, -1) for z in ns], dtype=np.int64)
        valid = (src_idx >= 0) & (tgt_idx >= 0)
        src_idx = src_idx[valid]
        tgt_idx = tgt_idx[valid]

        if len(src_idx) == 0:
            log.warning("No adjacency edges resolved — using identity")
            idx = torch.arange(n)
            return torch.sparse_coo_tensor(
                torch.stack([idx, idx]), torch.ones(n), (n, n)
            ).coalesce()

        rows_t = torch.from_numpy(src_idx)
        cols_t = torch.from_numpy(tgt_idx)
        vals_t = torch.ones(len(rows_t), dtype=torch.float32)

        A = torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t]), vals_t, (n, n)
        ).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
        norm_vals = vals_t / deg[rows_t]

        return torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t]), norm_vals, (n, n)
        ).coalesce()

    def _gnn_forward(self, X_t, adj_norm):
        """Reconstruct GraphSAGE from JSON weights and run forward pass.

        Architecture (matches train_and_export_gnn_v2.py):
            encoder: Linear(n_feat, 64) + LayerNorm(64) + ReLU
            sage1:   SAGEConv(64, 64) = Linear(128, 64) on cat(h, neigh_mean) + LayerNorm + ReLU
            sage2:   SAGEConv(64, 32) = Linear(128, 32) on cat(h, neigh_mean)
        """
        import torch
        import torch.nn.functional as F

        w = self._gnn_weights

        # Encoder: Linear + LayerNorm + ReLU
        enc_w = torch.tensor(w["encoder_weight"], dtype=torch.float32)
        enc_b = torch.tensor(w["encoder_bias"], dtype=torch.float32)
        ln_w = torch.tensor(w["encoder_ln_weight"], dtype=torch.float32)
        ln_b = torch.tensor(w["encoder_ln_bias"], dtype=torch.float32)

        h = X_t @ enc_w.T + enc_b
        h = F.layer_norm(h, [h.shape[-1]], ln_w, ln_b)
        h = F.relu(h)

        # SAGEConv1: cat(h, neigh_mean) -> Linear(128, 64) + LayerNorm + ReLU
        conv1_w = torch.tensor(w["conv1_weight"], dtype=torch.float32)
        conv1_b = torch.tensor(w["conv1_bias"], dtype=torch.float32)
        norm1_w = torch.tensor(w["norm1_weight"], dtype=torch.float32)
        norm1_b = torch.tensor(w["norm1_bias"], dtype=torch.float32)

        neigh1 = torch.sparse.mm(adj_norm, h)
        h = torch.cat([h, neigh1], dim=-1) @ conv1_w.T + conv1_b
        h = F.layer_norm(h, [h.shape[-1]], norm1_w, norm1_b)
        h = F.relu(h)

        # SAGEConv2: cat(h, neigh_mean) -> Linear(128, 32)
        conv2_w = torch.tensor(w["conv2_weight"], dtype=torch.float32)
        conv2_b = torch.tensor(w["conv2_bias"], dtype=torch.float32)

        neigh2 = torch.sparse.mm(adj_norm, h)
        h = torch.cat([h, neigh2], dim=-1) @ conv2_w.T + conv2_b

        return h

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Apply StandardScaler + PCA projection (PCA models only).

        Args:
            x: (n_features,) or (n_samples, n_features) raw ACS features
        Returns:
            (n_components,) or (n_samples, n_components) projected features
        """
        if self._is_gnn:
            raise RuntimeError(
                "GNN models use precomputed latents from set_graph(), "
                "not per-sample projection. Use predict_batch with set_graph() first."
            )
        x = np.asarray(x, dtype=np.float64)
        z = (x - self._means) / self._stds
        return z @ self._components.T

    def _predict_gbdt(self, z: np.ndarray, task: str) -> float:
        """Run GBDT tree walker on projected features.

        Args:
            z: (n_components,) PCA-projected feature vector
            task: task name (e.g. 'diabetes')
        Returns:
            Predicted prevalence (%)
        """
        gbdt = self._artifact["predictors"][task]
        pred = gbdt["init"]
        lr = gbdt["lr"]
        for tree in gbdt["trees"]:
            pred += lr * _walk(tree, z)
        return float(pred)

    def predict(self, x_raw: np.ndarray, task: str) -> float:
        """Predict a single task for a single ZCTA.

        PCA: projects x_raw directly. GNN: looks up precomputed latent by row index.

        Args:
            x_raw: (n_features,) ACS feature values in feature_order
                   For GNN, pass the row index (int) instead.
            task: task name (e.g. 'diabetes', 'obesity')
        Returns:
            Predicted prevalence (%)
        """
        if task not in self._artifact["predictors"]:
            raise ValueError(
                f"Unknown task: {task}. Available: {self._tasks}"
            )
        if self._is_gnn:
            if self._latent_cache is None:
                raise RuntimeError("GNN model: call set_graph() before predict")
            if isinstance(x_raw, (int, np.integer)):
                z = self._latent_cache[x_raw]
            else:
                raise TypeError(
                    "GNN predict() expects a row index (int). "
                    "Use predict_batch() with the full feature matrix after set_graph()."
                )
        else:
            z = self._project(x_raw)
        return self._predict_gbdt(z, task)

    def predict_all(self, x_raw: np.ndarray) -> dict:
        """Predict all 27 tasks for a single ZCTA.

        Args:
            x_raw: (n_features,) ACS feature values (PCA) or row index (GNN)
        Returns:
            {task_name: predicted_prevalence} for all tasks
        """
        if self._is_gnn:
            if self._latent_cache is None:
                raise RuntimeError("GNN model: call set_graph() before predict_all")
            if isinstance(x_raw, (int, np.integer)):
                z = self._latent_cache[x_raw]
            else:
                raise TypeError("GNN predict_all() expects a row index (int)")
        else:
            z = self._project(x_raw)
        return {task: self._predict_gbdt(z, task) for task in self._tasks}

    def predict_batch(
        self, X: np.ndarray, task: str
    ) -> np.ndarray:
        """Predict a single task for multiple ZCTAs.

        Args:
            X: (n_samples, n_features) ACS feature matrix
               For GNN models, X is ignored — uses precomputed latents from set_graph().
            task: task name
        Returns:
            (n_samples,) predicted prevalences
        """
        if task not in self._artifact["predictors"]:
            raise ValueError(f"Unknown task: {task}")
        if self._is_gnn:
            if self._latent_cache is None:
                raise RuntimeError("GNN model: call set_graph() before predict_batch")
            Z = self._latent_cache
        else:
            Z = self._project(X)
        gbdt = self._artifact["predictors"][task]
        preds = np.full(len(Z), gbdt["init"])
        lr = gbdt["lr"]
        for tree in gbdt["trees"]:
            for i in range(len(Z)):
                preds[i] += lr * _walk(tree, Z[i])
        return preds

    def predict_batch_all(self, X: np.ndarray = None) -> dict:
        """Predict all tasks for multiple ZCTAs.

        Args:
            X: (n_samples, n_features) ACS feature matrix.
               For GNN models, X is ignored — uses precomputed latents from set_graph().
        Returns:
            {task_name: (n_samples,) array} for all tasks
        """
        if self._is_gnn:
            if self._latent_cache is None:
                raise RuntimeError("GNN model: call set_graph() before predict_batch_all")
            Z = self._latent_cache
        else:
            if X is None:
                raise ValueError("PCA models require X (feature matrix)")
            Z = self._project(X)
        results = {}
        for task in self._tasks:
            gbdt = self._artifact["predictors"][task]
            preds = np.full(len(Z), gbdt["init"])
            lr = gbdt["lr"]
            for tree in gbdt["trees"]:
                for i in range(len(Z)):
                    preds[i] += lr * _walk(tree, Z[i])
            results[task] = preds
        return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main():
    """Demo: load model, score 5 random ZCTAs from the training data."""
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = GeoCertModel.from_s3("geo_cert_pca32_v1")
    print(f"\nModel: geo_cert_pca32_v1")
    print(f"Tasks: {len(model.tasks)}")
    print(f"Features: {model.n_features} -> {model.n_components}D")
    print(f"SHA256: {model.sha256[:16]}...")

    # Load real data for demo
    try:
        df = pd.read_parquet("C:/tmp/geo_data/zcta_features_labels.parquet")
    except FileNotFoundError:
        s3 = boto3.client("s3", region_name=REGION)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        s3.download_file(
            "yrsn-datasets",
            "rsct_curriculum/series_018/processed/zcta_features_labels.parquet",
            tmp.name,
        )
        df = pd.read_parquet(tmp.name)

    sample = df.sample(5, random_state=42)
    demo_tasks = ["diabetes", "obesity", "high_blood_pressure", "mental_health_not_good", "tree_cover"]

    print(f"\n{'ZCTA':<8} {'State':<6}", end="")
    for t in demo_tasks:
        print(f" {t[:12]:>12}", end="")
    print()
    print("-" * 80)

    for _, row in sample.iterrows():
        x = row[model.feature_order].values.astype(np.float64)
        x = np.nan_to_num(x, nan=0.0)
        print(f"{row['zcta_id']:<8} {row['state']:<6}", end="")
        for t in demo_tasks:
            pred = model.predict(x, t)
            actual = row.get(f"target_{t}", float("nan"))
            print(f" {pred:>5.1f}/{actual:>5.1f}", end="")
        print()

    print("\nFormat: predicted/actual (%)")


if __name__ == "__main__":
    main()
