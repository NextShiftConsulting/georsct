#!/usr/bin/env python3
"""
quickstart.py -- Download GeoRSCT and verify you have good data.

Run this once after cloning or downloading the dataset. It checks file
integrity, prints a summary of what you have, and runs a toy baseline
so you know the splits work end-to-end.

Usage:
    python quickstart.py                          # if files are in current dir
    python quickstart.py --data-dir /path/to/dir  # point to parquet location

Prerequisites:
    pip install pandas pyarrow scikit-learn
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def section(title: str) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print()


def main():
    parser = argparse.ArgumentParser(description="GeoRSCT quickstart: verify and summarize")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing geocert parquet files")
    args = parser.parse_args()

    data = Path(args.data_dir)
    table_path = data / "georsct_table.parquet"
    checksum_path = data / "georsct_checksums.sha256"
    errors = []

    # ==================================================================
    section("1. FILE CHECK")
    # ==================================================================

    expected_files = [
        "georsct_table.parquet",
        "georsct_simplified_001.geoparquet",
        "georsct_schema.json",
        "build_manifest.json",
        "georsct_checksums.sha256",
        "load_georsct.py",
    ]

    for f in expected_files:
        p = data / f
        if p.exists():
            size = p.stat().st_size / (1024 * 1024)
            print(f"  [OK]   {f:50s} {size:8.1f} MB")
        else:
            print(f"  [MISS] {f}")
            if f == "geocert_table.parquet":
                errors.append(f"Missing required file: {f}")

    # ==================================================================
    section("2. CHECKSUM VERIFICATION")
    # ==================================================================

    if checksum_path.exists():
        expected = {}
        for line in checksum_path.read_text().strip().split("\n"):
            h, name = line.split("  ", 1)
            expected[name] = h

        for name, expect_hash in sorted(expected.items()):
            p = data / name
            if not p.exists():
                print(f"  [SKIP] {name} (not found)")
                continue
            actual = sha256_file(p)
            if actual == expect_hash:
                print(f"  [OK]   {name}")
            else:
                print(f"  [FAIL] {name}")
                print(f"         expected: {expect_hash[:24]}...")
                print(f"         got:      {actual[:24]}...")
                errors.append(f"Checksum mismatch: {name}")
    else:
        print("  [SKIP] No checksums file found")

    # ==================================================================
    section("3. DATA SUMMARY")
    # ==================================================================

    if not table_path.exists():
        print("  Cannot load data -- geocert_table.parquet not found.")
        print()
        print("ERRORS:", errors)
        sys.exit(1)

    df = pd.read_parquet(table_path)
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    feat_cols = sorted(c for c in df.columns if c.startswith("acs_"))
    tgt_cols = sorted(c for c in df.columns if c.startswith("target_"))

    print(f"  Rows:     {len(df):,}")
    print(f"  Columns:  {len(df.columns)}")
    print(f"  Features: {len(feat_cols)} (acs_*)")
    print(f"  Targets:  {len(tgt_cols)} (target_*)")
    print()

    # Row count check
    if len(df) != 31789:
        errors.append(f"Expected 31,789 rows, got {len(df)}")
        print(f"  [FAIL] Row count: {len(df)} (expected 31,789)")
    else:
        print(f"  [OK]   Row count: 31,789")

    # Unique IDs
    if df["zcta_id"].is_unique:
        print(f"  [OK]   ZCTA IDs: unique")
    else:
        errors.append("Duplicate ZCTA IDs")
        print(f"  [FAIL] ZCTA IDs: duplicates found")

    # No bogus negatives in ACS
    bad_acs = {}
    for col in feat_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                bad_acs[col] = n_neg
    if bad_acs:
        print(f"  [FAIL] ACS features: {sum(bad_acs.values())} bogus negatives in {len(bad_acs)} columns")
        for col, n in bad_acs.items():
            print(f"           {col}: {n}")
        errors.append("Bogus negative ACS values -- run clean_acs_negatives.py")
    else:
        print(f"  [OK]   ACS features: no bogus negatives")

    # ==================================================================
    section("4. COVERAGE")
    # ==================================================================

    flags = ["has_cdc_places", "has_income", "has_home_value"]
    for flag in flags:
        n_true = int(df[flag].sum())
        n_false = len(df) - n_true
        pct = n_true / len(df) * 100
        print(f"  {flag:20s}  True: {n_true:,} ({pct:.1f}%)   False: {n_false:,}")

    # ==================================================================
    section("5. TARGET SUMMARY")
    # ==================================================================

    print(f"  {'Target':<45s} {'NaN':>6s}  {'Min':>10s}  {'Max':>10s}  {'Mean':>10s}")
    print(f"  {'-'*45} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for col in tgt_cols:
        n_nan = int(df[col].isna().sum())
        vals = df[col].dropna()
        print(f"  {col:<45s} {n_nan:>6d}  {vals.min():>10.2f}  {vals.max():>10.2f}  {vals.mean():>10.2f}")

    # ==================================================================
    section("6. SPLIT SANITY CHECK")
    # ==================================================================

    protocols = {
        "split_imputation": {"folds": ["valid1", "valid2", "valid3", "valid4", "valid5", "test"]},
        "split_extrapolation": {"folds": ["valid1", "valid2", "valid3", "valid4", "test"]},
        "split_superres": {"folds": ["valid", "test"]},
    }

    for col, info in protocols.items():
        print(f"  {col}:")
        vc = df[col].value_counts().sort_index()
        for val, cnt in vc.items():
            marker = "*" if val in info["folds"] else " "
            print(f"    {marker} {val:12s}  {cnt:,} ZCTAs")

        # Check no NaN
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            errors.append(f"{col} has {n_nan} NaN values")
            print(f"    [FAIL] {n_nan} NaN values")
        print()

    # ==================================================================
    section("7. TOY BASELINE (Ridge Regression)")
    # ==================================================================

    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        import numpy as np

        # Use imputation protocol, fold 1, predict diabetes
        target = "target_diabetes"
        split_col = "split_imputation"

        complete = df[df["has_cdc_places"] & df[feat_cols].notna().all(axis=1)].copy()

        train = complete[~complete[split_col].isin(["valid1", "test"])]
        val = complete[complete[split_col] == "valid1"]
        test = complete[complete[split_col] == "test"]

        X_train = train[feat_cols].values
        y_train = train[target].values
        X_val = val[feat_cols].values
        y_val = val[target].values
        X_test = test[feat_cols].values
        y_test = test[target].values

        # Impute NaN features with column mean
        col_means = np.nanmean(X_train, axis=0)
        for X in [X_train, X_val, X_test]:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                X[mask, i] = col_means[i]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        r2_val = r2_score(y_val, model.predict(X_val))
        r2_test = r2_score(y_test, model.predict(X_test))

        print(f"  Task:     {target}")
        print(f"  Protocol: imputation (fold 1)")
        print(f"  Model:    Ridge(alpha=1.0)")
        print(f"  Train:    {len(train):,} ZCTAs")
        print(f"  Val:      {len(val):,} ZCTAs  ->  R2 = {r2_val:.4f}")
        print(f"  Test:     {len(test):,} ZCTAs  ->  R2 = {r2_test:.4f}")
        print()

        if r2_val > 0.3:
            print(f"  [OK]   Baseline R2 > 0.3 -- data, splits, and features work end-to-end")
        else:
            print(f"  [WARN] Low R2 -- expected > 0.3 for ridge on diabetes")

    except ImportError:
        print("  [SKIP] scikit-learn not installed (pip install scikit-learn)")
        print("         Skipping toy baseline -- data checks above are sufficient.")

    # ==================================================================
    section("RESULT")
    # ==================================================================

    if errors:
        print(f"  ISSUES FOUND: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
        print()
        print("  Fix the issues above before using this data.")
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED")
        print()
        print("  You're ready to go. Quick start:")
        print()
        print("    from load_geocert import load_geocert, get_split")
        print()
        print('    df = load_geocert("geocert_table.parquet")')
        print('    train, val, test = get_split(df, protocol="imputation", fold=1)')
        print()
        print('    X_train = train[[c for c in train.columns if c.startswith("acs_")]]')
        print('    y_train = train["target_diabetes"]')


if __name__ == "__main__":
    main()
