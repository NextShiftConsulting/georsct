#!/usr/bin/env python3
"""
generate_oof_from_artifacts.py — OOF predictions from production artifacts.

Loads trained models via GeoCertModel.from_s3() and generates test-set
predictions. Uses the exact same weights that ship in production — no
retraining, no seed variance, deterministic output.

This closes the certificate chain gap: artifact SHA → OOF → YRSNCertificate.

Output: one ceiling_schema parquet per model, identical format to
generate_oof_predictions.py but with artifact provenance.

Usage:
    python -m apps.geo_cert.models.ceiling.generate_oof_from_artifacts
    python -m apps.geo_cert.models.ceiling.generate_oof_from_artifacts --models pca_v1,gnn_v2
    python -m apps.geo_cert.models.ceiling.generate_oof_from_artifacts --output-dir C:/tmp/oof_artifacts
"""

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from apps.geo_cert.inference import GeoCertModel
from apps.geo_cert.models.ceiling.ceiling_schema import (
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

# Model name → (artifact key for GeoCertModel, OOF model_version label)
# Note: geo_cert_gnn_v1 artifact contains v2 data (27 tasks, raw latents)
# because v2 job overwrote v1 filenames. See s018e README.
MODEL_CONFIGS = {
    "pca_v1": ("geo_cert_pca32_v1", "pca_v1"),
    "spatial_lag_v1": ("geo_cert_spatial_lag_v1", "spatial_lag_v1"),
    "gnn_v2": ("geo_cert_gnn_v1", "gnn_v2"),
}

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
    "AK": "02", "HI": "15", "PR": "72", "VI": "78", "GU": "66",
    "AS": "60", "MP": "69",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(local_path: str = None) -> pd.DataFrame:
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


def get_state_fips(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        STATE_ABBREV_TO_FIPS.get(s, "99") for s in df["state"].values
    ])


# ---------------------------------------------------------------------------
# Spatial lag computation
# ---------------------------------------------------------------------------

def compute_spatial_lags(
    df: pd.DataFrame,
    feature_cols: list,
    adjacency: dict,
    id_col: str = "zcta_id",
) -> pd.DataFrame:
    """Compute neighbor-mean spatial lags for each feature column.

    For each ZCTA, lag_acs_X = mean(acs_X) across geographic neighbors.
    Mirrors the logic in train_and_export_spatial_lag.py.

    Args:
        df: DataFrame with zcta_id and feature columns.
        feature_cols: list of column names (e.g. acs_* columns).
        adjacency: {zcta_str: [neighbor_zcta_str, ...]} dict.
        id_col: column with ZCTA identifiers.

    Returns:
        df with lag_{col} columns appended.
    """
    ids = df[id_col].values
    id_to_idx = {str(z).zfill(5): i for i, z in enumerate(ids)}

    X = df[feature_cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = medians[j]

    lag_X = np.zeros_like(X)
    n_with = 0
    for i, z in enumerate(ids):
        z5 = str(z).zfill(5)
        nbrs = adjacency.get(z5, [])
        valid = [id_to_idx[n] for n in nbrs if n in id_to_idx]
        if valid:
            lag_X[i] = X[valid].mean(axis=0)
            n_with += 1

    log.info(f"Spatial lags: {n_with}/{len(ids)} ZCTAs with neighbors")
    return df.assign(**{f"lag_{c}": lag_X[:, j] for j, c in enumerate(feature_cols)})


def _adj_df_to_dict(adj_df: pd.DataFrame) -> dict:
    """Convert adjacency edge-list DataFrame to {zcta: [neighbors]} dict."""
    adjacency = {}
    for _, row in adj_df.iterrows():
        src = str(int(row.iloc[0])).zfill(5)
        dst = str(int(row.iloc[1])).zfill(5)
        adjacency.setdefault(src, []).append(dst)
    return adjacency


def _needs_spatial_lags(model: "GeoCertModel") -> bool:
    """Check if a model's feature_order includes lag_acs_* columns."""
    return any(f.startswith("lag_") for f in model.feature_order)


# ---------------------------------------------------------------------------
# OOF generation from artifact
# ---------------------------------------------------------------------------

def generate_oof_for_model(
    model: GeoCertModel,
    model_version: str,
    df: pd.DataFrame,
    test_mask: np.ndarray,
    zcta_ids: np.ndarray,
    state_fips: np.ndarray,
) -> pd.DataFrame:
    """Generate predictions for ALL samples using a loaded artifact.

    Produces predictions for both train and test splits. Test predictions
    are true OOF (model never saw test data). Train predictions are
    in-sample but needed for s018h tercile label construction.

    For PCA models: project features and run GBDT.
    For GNN models: latents must already be set via set_graph().
    """
    from sklearn.metrics import r2_score

    all_rows = []
    train_mask = ~test_mask

    for task in SUPPORTED_TASK_NAMES:
        target_col = f"target_{task}"
        if target_col not in df.columns:
            continue
        if task not in model.tasks:
            continue

        y = df[target_col].values.astype(np.float64)
        valid = ~np.isnan(y)
        mask = valid  # predict all valid samples

        if mask.sum() == 0:
            continue

        if model.is_gnn:
            preds_all = model.predict_batch(None, task)
            y_pred = preds_all[mask]
        else:
            X = df.loc[mask, model.feature_order].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0)
            y_pred = model.predict_batch(X, task)

        y_true = y[mask]

        # Assign fold labels
        folds = np.where(test_mask[mask], "test", "train")

        # Report R² on test split only
        te = test_mask[mask]
        if te.sum() > 0:
            r2 = r2_score(y_true[te], y_pred[te])
            log.info(f"    {task}: R²={r2:.4f} ({te.sum()} test, {(~te).sum()} train)")

        rows = build_oof_rows(
            zctas=zcta_ids[mask],
            task=task,
            fold=folds,
            y_true=y_true,
            y_pred=y_pred,
            model_version=model_version,
            state_fips=state_fips[mask],
        )
        all_rows.append(rows)

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(
    local_data: str = None,
    output_dir: str = "C:/tmp/oof_artifacts",
    models: set = None,
):
    if models is None:
        models = set(MODEL_CONFIGS.keys())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(local_data)
    test_mask = (df["split_imputation"] == "test").values
    zcta_ids = df["zcta_id"].values
    state_fips = get_state_fips(df)

    log.info(f"Data: {len(df)} rows, test={test_mask.sum()}")
    log.info(f"Models: {sorted(models)}")

    # Load adjacency if any GNN or spatial_lag model requested
    adj_df = None
    needs_adj = any(
        MODEL_CONFIGS[m][0].startswith("geo_cert_gnn") or "spatial_lag" in m
        for m in models if m in MODEL_CONFIGS
    )
    if needs_adj:
        adj_df = load_adjacency(local_data)
        log.info(f"Adjacency loaded: {len(adj_df)} edges")

    results = {}
    provenance = {}

    for model_key in sorted(models):
        if model_key not in MODEL_CONFIGS:
            log.warning(f"Unknown model: {model_key}, skipping")
            continue

        artifact_name, version_label = MODEL_CONFIGS[model_key]
        log.info(f"\n{'=' * 60}")
        log.info(f"Loading {artifact_name} from S3...")

        model = GeoCertModel.from_s3(artifact_name)
        log.info(f"  SHA256: {model.sha256[:16]}...")
        log.info(f"  Tasks: {len(model.tasks)}")
        log.info(f"  Features: {model.n_features} -> {model.n_components}D")
        log.info(f"  Type: {'GNN' if model.is_gnn else 'PCA'}")

        # GNN models need graph setup
        if model.is_gnn:
            X_all = df[model.feature_order].values.astype(np.float64)
            X_all = np.nan_to_num(X_all, nan=0.0)
            model.set_graph(X_all, adj_df, zcta_ids)

        # Spatial lag models need lag_acs_* columns computed from adjacency
        df_model = df
        if _needs_spatial_lags(model) and adj_df is not None:
            acs_cols = sorted(
                c for c in df.columns
                if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
            )
            adjacency_dict = _adj_df_to_dict(adj_df)
            df_model = compute_spatial_lags(df, acs_cols, adjacency_dict)
            log.info(f"  Computed {len(acs_cols)} spatial lag features")

        oof = generate_oof_for_model(
            model, version_label, df_model, test_mask, zcta_ids, state_fips,
        )

        path = out / f"oof_{model_key}.parquet"
        oof.to_parquet(path, index=False)
        results[model_key] = (path, len(oof))

        provenance[model_key] = {
            "artifact_name": artifact_name,
            "sha256": model.sha256,
            "n_tasks": len(model.tasks),
            "n_features": model.n_features,
            "n_components": model.n_components,
            "is_gnn": model.is_gnn,
            "oof_rows": len(oof),
            "oof_path": str(path),
        }

        log.info(f"  Written {path}: {len(oof)} rows")

    # Write provenance manifest
    prov_path = out / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    log.info(f"\nProvenance: {prov_path}")

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("OOF FROM ARTIFACTS — SUMMARY")
    log.info("=" * 60)
    for mv, (path, n_rows) in sorted(results.items()):
        sha = provenance[mv]["sha256"][:16]
        log.info(f"  {mv}: {n_rows} rows, SHA={sha}... -> {path}")
    log.info(f"Total: {sum(n for _, n in results.values())} rows across {len(results)} models")

    # Validate all outputs
    for mv, (path, _) in results.items():
        oof = pd.read_parquet(path)
        errors = validate(oof, strict=False)
        if errors:
            log.error(f"  VALIDATION FAIL {mv}: {errors}")
        else:
            log.info(f"  VALIDATION PASS {mv}")

    return results, provenance


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOF predictions from production artifacts (no retraining)."
    )
    parser.add_argument(
        "--local-data", type=str, default="C:/tmp/geo_data",
        help="Local data directory (default: C:/tmp/geo_data)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="C:/tmp/oof_artifacts",
        help="Output directory for parquets (default: C:/tmp/oof_artifacts)"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model list (default: all). "
             "Options: pca_v1,spatial_lag_v1,gnn_v2"
    )
    args = parser.parse_args()

    models = None
    if args.models:
        models = set(args.models.split(","))
        invalid = models - set(MODEL_CONFIGS.keys())
        if invalid:
            parser.error(f"Unknown models: {invalid}. Valid: {sorted(MODEL_CONFIGS.keys())}")

    generate_all(
        local_data=args.local_data,
        output_dir=args.output_dir,
        models=models,
    )


if __name__ == "__main__":
    main()
