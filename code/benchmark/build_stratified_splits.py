#!/usr/bin/env python3
"""
build_stratified_splits.py -- Build stratified evaluation splits for GeoCert v24.

Replaces the unstratified random county partition from PDFM with
county-blocked, multi-dimensional stratified splits.

Block key: (state, county_fips) -- ~3,211 blocks nationally.

Stratification dimensions (balanced across folds):
  1. SVI quartile (vulnerability)
  2. Urban/rural (population tertile)
  3. Hospital access (nearest hospital > 50km flag)

Three evaluation protocols:
  - Imputation:       County-blocked 5-fold CV + test (geographic interpolation)
  - Extrapolation:    State-blocked 4-fold CV + test (distribution shift)
  - Super-resolution: County-aggregated train -> ZCTA-level prediction

Output: geocert_splits_v24.parquet
  - zcta_id
  - county_fips
  - state
  - split_imputation     (valid1..valid5, test)
  - split_extrapolation  (valid1..valid4, test)
  - split_superres       (train, test)
  - strat_svi_quartile   (Q1..Q4)
  - strat_urban_rural    (urban, suburban, rural)
  - strat_hospital_access (accessible, remote)

Usage:
    python build_stratified_splits.py --output /tmp/geocert_splits_v24.parquet
    python build_stratified_splits.py --upload
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SEED = 42

# Extrapolation test states: same 10 as PDFM for comparability
EXTRAP_TEST_STATES = {"CA", "MA", "ME", "NM", "NV", "NY", "SC", "TN", "WA", "WY"}

# Super-resolution: 20% of counties held out at ZCTA level
SUPERRES_TEST_FRACTION = 0.20


def build_strat_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stratification bins for each ZCTA."""
    result = df[["zcta_id", "county_fips", "state"]].copy()

    # 1. SVI quartile
    if "svi_overall" in df.columns:
        valid_svi = df["svi_overall"].notna()
        result["strat_svi_quartile"] = "Q_missing"
        result.loc[valid_svi, "strat_svi_quartile"] = pd.qcut(
            df.loc[valid_svi, "svi_overall"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
        ).astype(str)
    else:
        result["strat_svi_quartile"] = "Q_unavail"

    # 2. Urban/rural (population tertiles)
    pop_col = None
    for col in ["acs_total_pop", "population", "totalpopulation"]:
        if col in df.columns:
            pop_col = col
            break

    if pop_col:
        valid_pop = df[pop_col].notna() & (df[pop_col] > 0)
        result["strat_urban_rural"] = "unknown"
        result.loc[valid_pop, "strat_urban_rural"] = pd.qcut(
            df.loc[valid_pop, pop_col], 3, labels=["rural", "suburban", "urban"]
        ).astype(str)
    else:
        result["strat_urban_rural"] = "unknown"

    # 3. Hospital access
    if "hifld_nearest_hospital_km" in df.columns:
        result["strat_hospital_access"] = np.where(
            df["hifld_nearest_hospital_km"] > 50, "remote", "accessible"
        )
        result.loc[df["hifld_nearest_hospital_km"].isna(), "strat_hospital_access"] = "unknown"
    else:
        result["strat_hospital_access"] = "unknown"

    # Composite stratification key (for county-level balancing)
    result["strat_key"] = (
        result["strat_svi_quartile"] + "_" +
        result["strat_urban_rural"] + "_" +
        result["strat_hospital_access"]
    )

    return result


def county_majority_strat(strat_df: pd.DataFrame) -> pd.DataFrame:
    """Compute majority stratification bin for each county.

    Each county gets the modal (most common) strat_key among its ZCTAs.
    This is used to balance county assignment to folds.
    """
    county_strat = (
        strat_df.groupby("county_fips")["strat_key"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"strat_key": "county_strat_key"})
    )

    # Also track county state and size
    county_meta = (
        strat_df.groupby("county_fips")
        .agg(
            state=("state", "first"),
            n_zctas=("zcta_id", "size"),
        )
        .reset_index()
    )

    county_df = county_meta.merge(county_strat, on="county_fips")
    return county_df


def stratified_county_partition(
    county_df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> dict:
    """Assign counties to folds with stratification balance.

    Uses greedy bin-packing: for each stratum, distribute counties
    across folds to minimize imbalance in ZCTA counts.
    """
    rng = np.random.RandomState(seed)

    # Shuffle counties within each stratum
    county_df = county_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Track ZCTA count per fold
    fold_zcta_counts = np.zeros(n_folds, dtype=int)

    # Track stratum counts per fold (for balance checking)
    strata = county_df["county_strat_key"].unique()
    fold_strat_counts = {s: np.zeros(n_folds, dtype=int) for s in strata}

    # Assign each county to the fold with fewest ZCTAs in its stratum
    assignments = {}
    for _, row in county_df.iterrows():
        stratum = row["county_strat_key"]
        n_zctas = row["n_zctas"]

        # Find fold with fewest ZCTAs in this stratum (break ties by total count)
        strat_counts = fold_strat_counts[stratum]
        min_strat = strat_counts.min()
        candidates = np.where(strat_counts == min_strat)[0]

        if len(candidates) > 1:
            # Break tie by total ZCTA count
            best = candidates[np.argmin(fold_zcta_counts[candidates])]
        else:
            best = candidates[0]

        assignments[row["county_fips"]] = best
        fold_zcta_counts[best] += n_zctas
        fold_strat_counts[stratum][best] += n_zctas

    return assignments


def build_imputation_splits(
    strat_df: pd.DataFrame,
    county_df: pd.DataFrame,
    n_cv_folds: int = 5,
    test_fraction: float = 0.20,
    seed: int = SEED,
) -> pd.Series:
    """County-blocked stratified imputation splits.

    First reserves ~20% of counties for test, then splits remainder
    into n_cv_folds with stratified balance.
    """
    rng = np.random.RandomState(seed)

    # Step 1: Reserve test counties
    # Sample test counties stratified by county_strat_key
    test_counties = set()
    for stratum, group in county_df.groupby("county_strat_key"):
        n_test = max(1, int(round(len(group) * test_fraction)))
        sampled = group.sample(n=n_test, random_state=rng)
        test_counties.update(sampled["county_fips"])

    log.info("  Test counties: %d / %d (%.1f%%)",
             len(test_counties), len(county_df),
             len(test_counties) / len(county_df) * 100)

    # Step 2: Partition remaining counties into CV folds
    cv_counties = county_df[~county_df["county_fips"].isin(test_counties)].copy()
    fold_assignments = stratified_county_partition(cv_counties, n_cv_folds, seed)

    # Step 3: Map to ZCTAs
    split_col = pd.Series("unassigned", index=strat_df.index)

    for idx, row in strat_df.iterrows():
        cfips = row["county_fips"]
        if cfips in test_counties:
            split_col[idx] = "test"
        elif cfips in fold_assignments:
            fold = fold_assignments[cfips]
            split_col[idx] = f"valid{fold + 1}"
        else:
            split_col[idx] = "test"  # fallback

    return split_col


def build_extrapolation_splits(
    strat_df: pd.DataFrame,
    test_states: set,
    n_cv_folds: int = 4,
    seed: int = SEED,
) -> pd.Series:
    """State-blocked extrapolation splits.

    Test: pre-defined states (same as PDFM for comparability).
    CV: remaining states partitioned into folds, stratified by
    state-level characteristics.
    """
    rng = np.random.RandomState(seed)

    # Test split
    split_col = pd.Series("unassigned", index=strat_df.index)
    test_mask = strat_df["state"].isin(test_states)
    split_col[test_mask] = "test"

    # CV states
    cv_states = sorted(set(strat_df["state"].unique()) - test_states)
    rng.shuffle(cv_states)

    # Compute state-level population for balanced partitioning
    state_pop = strat_df[~test_mask].groupby("state").size().to_dict()

    # Greedy assignment by population balance
    fold_pops = np.zeros(n_cv_folds, dtype=int)
    state_fold = {}
    # Sort states by population descending (largest first for better balance)
    sorted_states = sorted(cv_states, key=lambda s: state_pop.get(s, 0), reverse=True)

    for state in sorted_states:
        best_fold = np.argmin(fold_pops)
        state_fold[state] = best_fold
        fold_pops[best_fold] += state_pop.get(state, 0)

    for idx, row in strat_df.iterrows():
        if row["state"] in state_fold:
            fold = state_fold[row["state"]]
            split_col[idx] = f"valid{fold + 1}"

    return split_col


def build_superres_splits(
    strat_df: pd.DataFrame,
    county_df: pd.DataFrame,
    test_fraction: float = SUPERRES_TEST_FRACTION,
    seed: int = SEED,
) -> pd.Series:
    """Super-resolution splits.

    Test counties are held out at ZCTA level.
    Train counties are aggregated to county level for training,
    but prediction is at ZCTA level for test counties.
    """
    rng = np.random.RandomState(seed)

    test_counties = set()
    for stratum, group in county_df.groupby("county_strat_key"):
        n_test = max(1, int(round(len(group) * test_fraction)))
        sampled = group.sample(n=n_test, random_state=seed + 7)  # different seed from imputation
        test_counties.update(sampled["county_fips"])

    split_col = pd.Series("train", index=strat_df.index)
    test_mask = strat_df["county_fips"].isin(test_counties)
    split_col[test_mask] = "test"

    return split_col


def validate_splits(df: pd.DataFrame):
    """Validate split quality: no county leaks, balanced strata."""
    log.info("")
    log.info("=== SPLIT VALIDATION ===")

    # 1. County blocking: no county in multiple folds
    for split_col in ["split_imputation", "split_extrapolation"]:
        county_folds = df.groupby("county_fips")[split_col].nunique()
        leaks = county_folds[county_folds > 1]
        if len(leaks) > 0:
            log.error("  LEAK in %s: %d counties in multiple folds!", split_col, len(leaks))
        else:
            log.info("  %s: county blocking OK (0 leaks)", split_col)

    # 2. State blocking for extrapolation
    state_folds = df.groupby("state")["split_extrapolation"].nunique()
    state_leaks = state_folds[state_folds > 1]
    if len(state_leaks) > 0:
        log.error("  LEAK in extrapolation: %d states in multiple folds!", len(state_leaks))
    else:
        log.info("  Extrapolation: state blocking OK (0 leaks)")

    # 3. Fold size balance
    for split_col in ["split_imputation", "split_extrapolation"]:
        fold_sizes = df[split_col].value_counts()
        log.info("  %s fold sizes:", split_col)
        for fold, size in fold_sizes.sort_index().items():
            log.info("    %s: %d ZCTAs (%.1f%%)", fold, size, size / len(df) * 100)

    # 4. Stratification balance (imputation only)
    log.info("")
    log.info("  Stratification balance (imputation):")
    for strat_col in ["strat_svi_quartile", "strat_urban_rural", "strat_hospital_access"]:
        if strat_col not in df.columns:
            continue
        ct = pd.crosstab(df["split_imputation"], df[strat_col], normalize="index")
        log.info("    %s:", strat_col)
        # Show max imbalance ratio
        if len(ct.columns) > 1:
            for col in ct.columns:
                vals = ct[col].values
                if vals.max() > 0:
                    ratio = vals.max() / max(vals.min(), 1e-6)
                    log.info("      %s: min=%.1f%% max=%.1f%% ratio=%.2f",
                             col, vals.min() * 100, vals.max() * 100, ratio)


def main():
    parser = argparse.ArgumentParser(description="Build stratified splits")
    parser.add_argument("--output", default="/tmp/geocert_splits_v24.parquet")
    parser.add_argument("--features", default="/tmp/zcta_features_labels.parquet",
                        help="ZCTA features file (needs zcta_id, population)")
    parser.add_argument("--crosswalk", default="/tmp/zcta_county_crosswalk.parquet",
                        help="ZCTA-county crosswalk")
    parser.add_argument("--svi", default=None,
                        help="SVI parquet (optional, for stratification)")
    parser.add_argument("--hifld", default=None,
                        help="HIFLD parquet (optional, for stratification)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--download-s3", action="store_true",
                        help="Download inputs from S3")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Load data
    if args.download_s3:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        s3 = boto3.client("s3", region_name="us-east-1")

        log.info("Downloading inputs from S3...")
        s3.download_file(BUCKET, "rsct_curriculum/series_018/processed/zcta_features_labels.parquet",
                         args.features)
        s3.download_file(BUCKET, "rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet",
                         args.crosswalk)

        svi_path = "/tmp/svi_zcta.parquet"
        try:
            s3.download_file(BUCKET, "rsct_curriculum/series_018/processed/svi_zcta.parquet", svi_path)
            args.svi = svi_path
        except Exception:
            log.warning("SVI data not available on S3")

        hifld_path = "/tmp/hifld_zcta.parquet"
        try:
            s3.download_file(BUCKET, "rsct_curriculum/series_018/processed/hifld_zcta.parquet", hifld_path)
            args.hifld = hifld_path
        except Exception:
            log.warning("HIFLD data not available on S3")

    # Load features
    log.info("Loading features from %s", args.features)
    features = pd.read_parquet(args.features)
    features["zcta_id"] = features["zcta_id"].astype(str).str.zfill(5)
    log.info("  %d ZCTAs, %d columns", len(features), len(features.columns))

    # Load crosswalk
    log.info("Loading crosswalk from %s", args.crosswalk)
    xwalk = pd.read_parquet(args.crosswalk)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)

    # Merge crosswalk -- drop existing state/county columns in favor of Census-authoritative
    for col in ["state", "county_name", "county_fips", "state_fips"]:
        if col in features.columns:
            features = features.drop(columns=[col])
    df = features.merge(xwalk[["zcta_id", "county_fips", "state"]], on="zcta_id", how="left")

    # Drop ZCTAs without county assignment
    missing_county = df["county_fips"].isna().sum()
    if missing_county > 0:
        log.warning("  %d ZCTAs without county assignment (dropped)", missing_county)
        df = df.dropna(subset=["county_fips"])

    log.info("  %d ZCTAs with county assignment", len(df))

    # Merge enrichment data for stratification
    if args.svi:
        svi = pd.read_parquet(args.svi)
        svi["zcta_id"] = svi["zcta_id"].astype(str).str.zfill(5)
        df = df.merge(svi[["zcta_id", "svi_overall"]], on="zcta_id", how="left")
        log.info("  SVI merged: %d non-null", df["svi_overall"].notna().sum())

    if args.hifld:
        hifld = pd.read_parquet(args.hifld)
        hifld["zcta_id"] = hifld["zcta_id"].astype(str).str.zfill(5)
        if "hifld_nearest_hospital_km" in hifld.columns:
            df = df.merge(hifld[["zcta_id", "hifld_nearest_hospital_km"]], on="zcta_id", how="left")
            log.info("  HIFLD merged: %d non-null", df["hifld_nearest_hospital_km"].notna().sum())

    # Build stratification bins
    log.info("Building stratification bins...")
    strat_df = build_strat_bins(df)

    # County-level stratification
    county_df = county_majority_strat(strat_df)
    log.info("  %d counties, %d strata", len(county_df), county_df["county_strat_key"].nunique())

    # Build splits
    log.info("")
    log.info("Building imputation splits (county-blocked, 5-fold + test)...")
    strat_df["split_imputation"] = build_imputation_splits(strat_df, county_df)

    log.info("Building extrapolation splits (state-blocked, 4-fold + test)...")
    strat_df["split_extrapolation"] = build_extrapolation_splits(strat_df, EXTRAP_TEST_STATES)

    log.info("Building super-resolution splits...")
    strat_df["split_superres"] = build_superres_splits(strat_df, county_df)

    # Add coverage flags from features
    for flag in ["has_cdc_places", "has_income", "has_home_value"]:
        if flag in features.columns:
            strat_df[flag] = features.set_index("zcta_id").loc[strat_df["zcta_id"].values, flag].values

    # Add has_cdc_ci if CI data exists
    ci_path = Path("/tmp/cdc_places_ci.parquet")
    if ci_path.exists():
        ci = pd.read_parquet(ci_path, columns=["zcta_id", "has_cdc_ci"])
        ci["zcta_id"] = ci["zcta_id"].astype(str).str.zfill(5)
        strat_df = strat_df.merge(ci, on="zcta_id", how="left")
        strat_df["has_cdc_ci"] = strat_df["has_cdc_ci"].fillna(False)

    # Validate
    validate_splits(strat_df)

    # Select output columns
    out_cols = [
        "zcta_id", "county_fips", "state",
        "split_imputation", "split_extrapolation", "split_superres",
        "strat_svi_quartile", "strat_urban_rural", "strat_hospital_access",
    ]
    # Add coverage flags if present
    for flag in ["has_cdc_places", "has_income", "has_home_value", "has_cdc_ci"]:
        if flag in strat_df.columns:
            out_cols.append(flag)

    out_cols = [c for c in out_cols if c in strat_df.columns]
    result = strat_df[out_cols].copy()

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:           %d", len(result))
    log.info("Counties:        %d", result["county_fips"].nunique())
    log.info("States:          %d", result["state"].nunique())
    log.info("")
    log.info("Imputation:      %s", dict(result["split_imputation"].value_counts().sort_index()))
    log.info("Extrapolation:   %s", dict(result["split_extrapolation"].value_counts().sort_index()))
    log.info("Super-res:       %s", dict(result["split_superres"].value_counts().sort_index()))
    log.info("")
    log.info("Strat SVI:       %s", dict(result["strat_svi_quartile"].value_counts().sort_index()))
    log.info("Strat urban:     %s", dict(result["strat_urban_rural"].value_counts().sort_index()))
    log.info("Strat hospital:  %s", dict(result["strat_hospital_access"].value_counts().sort_index()))

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/geocert_splits_v24.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "build_stratified_splits",
            "timestamp": timestamp,
            "seed": SEED,
            "n_zctas": len(result),
            "n_counties": int(result["county_fips"].nunique()),
            "n_states": int(result["state"].nunique()),
            "extrap_test_states": sorted(EXTRAP_TEST_STATES),
            "stratification_dims": ["svi_quartile", "urban_rural", "hospital_access"],
            "imputation_folds": dict(result["split_imputation"].value_counts().sort_index()),
            "extrapolation_folds": dict(result["split_extrapolation"].value_counts().sort_index()),
            "superres_folds": dict(result["split_superres"].value_counts().sort_index()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/geocert_splits_v24_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
