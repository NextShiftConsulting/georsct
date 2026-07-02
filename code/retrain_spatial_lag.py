#!/usr/bin/env python3
"""
Retrain spatial_lag_v1.npz on v23.0.2 columns only.

The original spatial_lag_v1.npz was trained on 80 features (33 ACS + 33 lag + 14 OSM/enrich)
from zcta_features_labels_with_lags.parquet. The published v23.0.2 georsct_table.parquet
has only 47 of those columns (33 ACS + 14 curated lag). This script retrains the
StandardScaler + PCA(32) transform on the 47 available columns so no zero-filling occurs.

Steps:
  1. Back up old artifact as spatial_lag_v1_80col.npz on S3
  2. Load georsct_table.parquet (v23.0.2)
  3. Select 33 acs_* + 14 lag_acs_* columns
  4. Fit StandardScaler + PCA(n_components=32, random_state=42)
  5. Save spatial_lag_v1.npz with keys:
     scaler_mean, scaler_scale, pca_components, pca_mean, feature_schema
  6. Upload to S3 (both v23 and legacy trees)
  7. Copy to local HF staging for upload

Usage:
  python retrain_spatial_lag.py                    # full run
  python retrain_spatial_lag.py --dry-run          # show what would happen
  python retrain_spatial_lag.py --local-only       # skip S3 upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BUCKET = "swarm-yrsn-datasets"
V23_PREFIX = "geocert/v23.0.2/representations"
LEGACY_PREFIX = "rsct_curriculum/series_018/artifacts/representations"

LOCAL_PARQUET = Path("C:/tmp/hf_upload/georsct_table.parquet")
FALLBACK_PARQUET = Path("C:/tmp/georsct_table.parquet")

OUTPUT_DIR = Path("C:/tmp/hf_upload/representations")

N_COMPONENTS = 32
RANDOM_STATE = 42


def load_parquet() -> pd.DataFrame:
    """Load v23.0.2 parquet from local staging."""
    for p in (LOCAL_PARQUET, FALLBACK_PARQUET):
        if p.exists():
            print(f"Loading parquet from {p}")
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"georsct_table.parquet not found at {LOCAL_PARQUET} or {FALLBACK_PARQUET}"
    )


def select_features(df: pd.DataFrame) -> list[str]:
    """Select the 33 acs_* + 14 lag_acs_* columns present in v23.0.2."""
    acs = sorted(c for c in df.columns if c.startswith("acs_"))
    lag = sorted(c for c in df.columns if c.startswith("lag_acs_"))
    cols = acs + lag
    print(f"Selected {len(acs)} ACS + {len(lag)} lag = {len(cols)} features")
    return cols


def fit_transform(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Fit StandardScaler + PCA(32) and return artifact dict."""
    X = df[feature_cols].values.astype(np.float64)

    # Median-impute NaN
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            med = np.nanmedian(X[:, j])
            X[nans, j] = med
            print(f"  Imputed {nans.sum()} NaN in {feature_cols[j]} (median={med:.4f})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA({N_COMPONENTS}) fitted: {explained:.1%} variance explained")

    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "feature_schema": np.array(feature_cols),
    }


def backup_old_on_s3(session, dry_run: bool = False):
    """Copy existing spatial_lag_v1.npz -> spatial_lag_v1_80col.npz on S3."""
    import botocore

    s3 = session.client("s3")
    for prefix in (V23_PREFIX, LEGACY_PREFIX):
        src_key = f"{prefix}/spatial_lag_v1.npz"
        dst_key = f"{prefix}/spatial_lag_v1_80col.npz"
        if dry_run:
            print(f"[DRY RUN] Would copy s3://{BUCKET}/{src_key} -> {dst_key}")
            continue
        try:
            s3.head_object(Bucket=BUCKET, Key=src_key)
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": src_key},
                Key=dst_key,
            )
            print(f"Backed up s3://{BUCKET}/{dst_key}")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"No existing artifact at s3://{BUCKET}/{src_key} (skip backup)")
            else:
                raise


def upload_to_s3(session, local_path: Path, dry_run: bool = False):
    """Upload new spatial_lag_v1.npz to both S3 trees."""
    s3 = session.client("s3")
    for prefix in (V23_PREFIX, LEGACY_PREFIX):
        key = f"{prefix}/spatial_lag_v1.npz"
        if dry_run:
            print(f"[DRY RUN] Would upload {local_path} -> s3://{BUCKET}/{key}")
        else:
            s3.upload_file(str(local_path), BUCKET, key)
            print(f"Uploaded s3://{BUCKET}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Retrain spatial_lag_v1.npz on v23.0.2 columns")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--local-only", action="store_true", help="Skip S3 backup/upload")
    args = parser.parse_args()

    # Load data
    df = load_parquet()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Select features
    feature_cols = select_features(df)

    if args.dry_run:
        print(f"\n[DRY RUN] Would fit StandardScaler + PCA({N_COMPONENTS}) on {len(feature_cols)} features")
        print(f"[DRY RUN] Would save to {OUTPUT_DIR / 'spatial_lag_v1.npz'}")
        if not args.local_only:
            print(f"[DRY RUN] Would backup old artifact as spatial_lag_v1_80col.npz on S3")
            print(f"[DRY RUN] Would upload to s3://{BUCKET}/{V23_PREFIX}/")
            print(f"[DRY RUN] Would upload to s3://{BUCKET}/{LEGACY_PREFIX}/")
        return

    # Fit
    artifact = fit_transform(df, feature_cols)

    # Save locally
    out_path = OUTPUT_DIR / "spatial_lag_v1.npz"
    np.savez_compressed(out_path, **artifact)
    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB)")

    # Verify round-trip
    check = np.load(out_path, allow_pickle=True)
    assert list(check["feature_schema"]) == feature_cols, "feature_schema mismatch"
    assert check["pca_components"].shape == (N_COMPONENTS, len(feature_cols)), "components shape mismatch"
    print(f"Verified: {len(feature_cols)} features, PCA {check['pca_components'].shape}")

    # S3 operations
    if not args.local_only:
        import boto3
        session = boto3.Session(profile_name="nsc-swarm", region_name="us-east-1")
        backup_old_on_s3(session)
        upload_to_s3(session, out_path)
    else:
        print("Skipping S3 upload (--local-only)")

    print("\nDone. Next steps:")
    print("  1. Upload to HuggingFace: huggingface-cli upload rudymartin/georsct representations/ representations/")
    print("  2. Re-run S019A/C/D experiments (spatial lag embedding values have changed)")


if __name__ == "__main__":
    main()
