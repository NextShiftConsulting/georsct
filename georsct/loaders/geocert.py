#!/usr/bin/env python3
"""
load_geocert.py -- Load and validate GeoCert benchmark data.

Copy this file alongside the parquet files. It handles:
  - Loading with or without geometry
  - Splitting into train/val/test for any evaluation protocol
  - Separating features (acs_*) from targets (target_*)
  - Filtering by coverage flags (has_cdc_places, has_income, has_home_value)
  - Basic data validation

Usage:
    from load_geocert import load_geocert, get_split

    # Load table (no geometry, fast)
    df = load_geocert("geocert_table.parquet")

    # Load with geometry
    geo = load_geocert("geocert_simplified_001.geoparquet")

    # Get train/val/test for imputation protocol, fold 1
    train, val, test = get_split(df, protocol="imputation", fold=1)

    # Get features and targets
    X_train = train[[c for c in train.columns if c.startswith("acs_")]]
    y_train = train["target_diabetes"]

    # Filter to ZCTAs with complete CDC PLACES labels
    complete = df[df["has_cdc_places"]]

Prerequisites:
    pip install pandas pyarrow        # table only
    pip install geopandas pyogrio     # with geometry
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_geocert(path: str) -> pd.DataFrame:
    """Load GeoCert parquet (table or geoparquet).

    Args:
        path: Path to geocert_table.parquet or geocert_simplified_001.geoparquet

    Returns:
        DataFrame (or GeoDataFrame if geoparquet)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if "geoparquet" in p.name:
        import geopandas as gpd
        df = gpd.read_parquet(p)
    else:
        df = pd.read_parquet(p)

    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Run basic integrity checks."""
    assert len(df) == 31789, f"Expected 31,789 rows, got {len(df)}"
    assert df["zcta_id"].is_unique, "Duplicate ZCTA IDs"

    # No bogus negatives in ACS features
    acs_cols = [c for c in df.columns if c.startswith("acs_")]
    for col in acs_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                raise ValueError(
                    f"{col} has {n_neg} negative values. "
                    f"Run clean_acs_negatives.py or replace negatives with NaN."
                )


def feature_columns(df: pd.DataFrame) -> list:
    """Return ACS feature column names."""
    return sorted(c for c in df.columns if c.startswith("acs_"))


def target_columns(df: pd.DataFrame) -> list:
    """Return target column names."""
    return sorted(c for c in df.columns if c.startswith("target_"))


def get_split(
    df: pd.DataFrame,
    protocol: str = "imputation",
    fold: int = 1,
    target: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Args:
        df: GeoCert DataFrame
        protocol: "imputation", "extrapolation", or "superres"
        fold: Validation fold number (1-5 for imputation, 1-4 for extrapolation,
              ignored for superres)
        target: If specified, drop rows where this target is NaN

    Returns:
        (train, val, test) DataFrames
    """
    col_map = {
        "imputation": "split_imputation",
        "extrapolation": "split_extrapolation",
        "superres": "split_superres",
    }

    if protocol not in col_map:
        raise ValueError(f"protocol must be one of {list(col_map.keys())}")

    split_col = col_map[protocol]

    if protocol == "superres":
        val_label = "valid"
    else:
        val_label = f"valid{fold}"
        valid_folds = {"imputation": 5, "extrapolation": 4}
        max_fold = valid_folds[protocol]
        if fold < 1 or fold > max_fold:
            raise ValueError(f"{protocol} has folds 1-{max_fold}, got {fold}")

    test = df[df[split_col] == "test"].copy()
    val = df[df[split_col] == val_label].copy()
    train = df[~df[split_col].isin([val_label, "test"])].copy()

    # Optionally filter to rows with non-null target
    if target:
        train = train[train[target].notna()]
        val = val[val[target].notna()]
        test = test[test[target].notna()]

    return train, val, test


def get_complete(df: pd.DataFrame, domain: str = "all") -> pd.DataFrame:
    """Filter to ZCTAs with complete labels for a domain.

    Args:
        domain: "health" (has_cdc_places), "income" (has_income),
                "home_value" (has_home_value), or "all" (all three)

    Returns:
        Filtered DataFrame
    """
    flag_map = {
        "health": ["has_cdc_places"],
        "income": ["has_income"],
        "home_value": ["has_home_value"],
        "all": ["has_cdc_places", "has_income", "has_home_value"],
    }
    if domain not in flag_map:
        raise ValueError(f"domain must be one of {list(flag_map.keys())}")

    mask = pd.Series(True, index=df.index)
    for flag in flag_map[domain]:
        mask &= df[flag]
    return df[mask].copy()


# -- Quick sanity check when run directly --
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "geocert_table.parquet"
    df = load_geocert(path)

    print(f"Loaded: {len(df)} ZCTAs, {len(df.columns)} columns")
    print(f"Features: {len(feature_columns(df))} ACS columns")
    print(f"Targets:  {len(target_columns(df))} target columns")
    print()

    # Coverage
    print("Coverage:")
    print(f"  has_cdc_places: {df['has_cdc_places'].sum()} / {len(df)}")
    print(f"  has_income:     {df['has_income'].sum()} / {len(df)}")
    print(f"  has_home_value: {df['has_home_value'].sum()} / {len(df)}")
    print()

    # Demo split
    train, val, test = get_split(df, protocol="imputation", fold=1, target="target_diabetes")
    print("Imputation split (fold 1, target=diabetes):")
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print()

    # NaN summary
    print("Target NaN counts:")
    for col in target_columns(df):
        n = df[col].isna().sum()
        if n > 0:
            print(f"  {col}: {n}")

    print()
    print("ACS feature NaN counts:")
    for col in feature_columns(df):
        n = df[col].isna().sum()
        if n > 0:
            print(f"  {col}: {n}")

    print()
    print("Validation passed.")
