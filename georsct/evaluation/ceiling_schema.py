"""
ceiling_schema.py — Shared OOF prediction schema for TRF estimation.

All ceiling models (PCA v1/v2, GNN v1/v2, kernel, MLP) emit predictions
in this format. The estimator validates on load; breaking the schema is
a test failure, not a silent column-drift bug.

State FIPS is included for block bootstrap (spatial autocorrelation).
Residual is precomputed to avoid recomputing per model in the audit.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Canonical column order and dtypes
OOF_COLUMNS = {
    "zcta": str,
    "task": str,
    "fold": str,
    "y_true": float,
    "y_pred": float,
    "residual": float,
    "model_version": str,
    "state_fips": str,
}

REQUIRED_COLUMNS = list(OOF_COLUMNS.keys())

# Pre-registered residual correlation threshold for flagging tasks
# where task residual floor estimate relies on limited architectural diversity.
# Set before seeing any correlation matrix. Do not revise.
RESIDUAL_CORRELATION_FLAG_THRESHOLD = 0.5

# Valid model version base identifiers.
# Seed variants (e.g. "gnn_v1_seed0", "mlp_v1_seed2") and shuffle variants
# (e.g. "mlp_shuffle_v1_seed0") are also valid — see _is_valid_model_version().
VALID_MODEL_VERSIONS = {
    "pca_v1", "pca_v2", "spatial_lag_v1",
    "gnn_v1", "gnn_v2",
    "kernel_v1", "mlp_v1",
    "mlp_shuffle_v1",
}


def _is_valid_model_version(version: str) -> bool:
    """Check if model version is valid, including seed variants."""
    if version in VALID_MODEL_VERSIONS:
        return True
    # Allow seed-suffixed variants: "gnn_v1_seed0", "mlp_v1_seed2", etc.
    import re
    base = re.sub(r"_seed\d+$", "", version)
    return base in VALID_MODEL_VERSIONS


@dataclass(frozen=True)
class OOFRow:
    """Single OOF prediction row."""
    zcta: str
    task: str
    fold: str
    y_true: float
    y_pred: float
    residual: float
    model_version: str
    state_fips: str


def validate(df: pd.DataFrame, strict: bool = True) -> list:
    """Validate OOF dataframe against schema.

    Args:
        df: DataFrame to validate
        strict: If True, raise on errors. If False, return error list.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors and strict:
        raise ValueError(f"OOF schema validation failed: {errors}")
        return errors

    # Check no nulls in required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            n_null = int(df[col].isna().sum())
            errors.append(f"Column '{col}' has {n_null} null values")

    # Check residual consistency (y_true - y_pred == residual)
    if all(c in df.columns for c in ("y_true", "y_pred", "residual")):
        computed = df["y_true"] - df["y_pred"]
        max_diff = (df["residual"] - computed).abs().max()
        if max_diff > 1e-6:
            errors.append(
                f"Residual inconsistency: max|residual - (y_true - y_pred)| = {max_diff:.2e}"
            )

    # Check model_version values (supports seed-suffixed variants)
    if "model_version" in df.columns:
        unknown = {
            v for v in df["model_version"].unique()
            if not _is_valid_model_version(v)
        }
        if unknown:
            errors.append(f"Unknown model_version values: {unknown}")

    # Check state_fips format (2-digit string, zero-padded)
    if "state_fips" in df.columns:
        bad_fips = df["state_fips"].str.len() != 2
        if bad_fips.any():
            n_bad = int(bad_fips.sum())
            errors.append(f"{n_bad} rows have state_fips not 2 chars")

    if strict and errors:
        raise ValueError(f"OOF schema validation failed: {errors}")

    return errors


def empty_oof_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with the correct schema."""
    return pd.DataFrame({col: pd.Series(dtype=dtype)
                         for col, dtype in OOF_COLUMNS.items()})


def build_oof_rows(
    zctas: list,
    task: str,
    fold: str,
    y_true,
    y_pred,
    model_version: str,
    state_fips: list,
) -> pd.DataFrame:
    """Build OOF rows from arrays.

    Args:
        zctas: ZCTA IDs
        task: Task name
        fold: Fold identifier (e.g. 'test', 'valid1')
        y_true: True target values (array-like)
        y_pred: Predicted values (array-like)
        model_version: Model identifier
        state_fips: State FIPS codes per ZCTA
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred

    return pd.DataFrame({
        "zcta": [str(z).zfill(5) for z in zctas],
        "task": task,
        "fold": fold,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residual,
        "model_version": model_version,
        "state_fips": [str(s).zfill(2) for s in state_fips],
    })
