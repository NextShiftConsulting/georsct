#!/usr/bin/env python3
"""
fetch_acs_moe.py -- Fetch ACS 5-Year Margin of Error (M suffix) at ZCTA level.

The Census API provides MOE as the M-suffix counterpart to E-suffix estimates.
For derived features (percentages), we propagate MOE using the Census Bureau's
approximation formula for ratios:

    MOE_proportion = sqrt(MOE_numerator^2 - (proportion^2 * MOE_denominator^2)) / denominator

When the inner term is negative (MOE_num < proportion * MOE_denom), use:
    MOE_proportion = sqrt(MOE_numerator^2 + (proportion^2 * MOE_denominator^2)) / denominator

Output: zcta_acs_margins_of_error.parquet
  - zcta_id (str, 5-digit zero-padded)
  - acs_{feature}_moe (float64) for each of 33 ACS features
  - has_acs_moe (bool) — True if all primary MOEs are present

Usage:
    export CENSUS_API_KEY="your_key"
    python fetch_acs_moe.py --output /tmp/zcta_acs_margins_of_error.parquet
    python fetch_acs_moe.py --upload   # also upload to S3
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Census API base URL -- ZCTA-level (no crosswalk needed)
ACS_API_BASE = "https://api.census.gov/data/2022/acs/acs5"

# State FIPS codes (50 states + DC)
STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# =============================================================================
# VARIABLE DEFINITIONS
# Each tuple: (E_var, M_var, feature_name, var_type, denominator_E, denominator_M)
#   var_type: "direct" = raw value, "ratio" = needs propagation
# =============================================================================

# Variables we need both E and M for
DIRECT_VARS: List[Tuple[str, str, str]] = [
    # (census_E_code, census_M_code, output_feature_name)
    ("B01001_001E", "B01001_001M", "acs_total_pop"),
    ("B01002_001E", "B01002_001M", "acs_median_age"),
    ("B25077_001E", "B25077_001M", "acs_median_home_value"),
    ("B25064_001E", "B25064_001M", "acs_median_rent"),
    ("B25035_001E", "B25035_001M", "acs_median_year_built"),
    ("B19013_001E", "B19013_001M", "acs_median_hh_income"),
    ("B19083_001E", "B19083_001M", "acs_gini_index"),
    ("B08303_001E", "B08303_001M", "acs_mean_commute_min"),
]

# Ratio vars: (num_E, num_M, denom_E, denom_M, output_feature_name)
RATIO_VARS: List[Tuple[str, str, str, str, str]] = [
    ("B02001_002E", "B02001_002M", "B02001_001E", "B02001_001M", "acs_pct_white"),
    ("B02001_003E", "B02001_003M", "B02001_001E", "B02001_001M", "acs_pct_black"),
    ("B02001_005E", "B02001_005M", "B02001_001E", "B02001_001M", "acs_pct_asian"),
    ("B03001_003E", "B03001_003M", "B03001_001E", "B03001_001M", "acs_pct_hispanic"),
    ("B15003_022E", "B15003_022M", "B15003_001E", "B15003_001M", "acs_pct_bachelors"),
    ("B09001_001E", "B09001_001M", "B01001_001E", "B01001_001M", "acs_pct_under_18"),
    ("B01001_026E", "B01001_026M", "B01001_001E", "B01001_001M", "acs_pct_female"),
    ("B21001_002E", "B21001_002M", "B21001_001E", "B21001_001M", "acs_pct_veterans"),
    ("B05001_006E", "B05001_006M", "B05001_001E", "B05001_001M", "acs_pct_foreign_born"),
    ("B06007_002E", "B06007_002M", "B06007_001E", "B06007_001M", "acs_pct_english_only"),
    ("B08301_003E", "B08301_003M", "B08301_001E", "B08301_001M", "acs_pct_drive_alone"),
    ("B08301_010E", "B08301_010M", "B08301_001E", "B08301_001M", "acs_pct_transit"),
    ("B08301_021E", "B08301_021M", "B08301_001E", "B08301_001M", "acs_pct_wfh"),
    ("B25003_002E", "B25003_002M", "B25003_001E", "B25003_001M", "acs_pct_owner_occupied"),
    ("B25003_003E", "B25003_003M", "B25003_001E", "B25003_001M", "acs_pct_renter_occupied"),
    ("B25002_003E", "B25002_003M", "B25002_001E", "B25002_001M", "acs_pct_vacant"),
    ("B17001_002E", "B17001_002M", "B17001_001E", "B17001_001M", "acs_pct_below_poverty"),
    ("B22010_002E", "B22010_002M", "B22010_001E", "B22010_001M", "acs_pct_food_stamps"),
    ("B23025_005E", "B23025_005M", "B23025_003E", "B23025_003M", "acs_unemployment_rate"),
]

# Composite ratio vars (sum of numerators / denominator)
# (num_E_list, num_M_list, denom_E, denom_M, output_feature_name)
COMPOSITE_VARS: List[Tuple[List[str], List[str], str, str, str]] = [
    # Graduate = masters + professional + doctorate / edu total
    (
        ["B15003_023E", "B15003_024E", "B15003_025E"],
        ["B15003_023M", "B15003_024M", "B15003_025M"],
        "B15003_001E", "B15003_001M",
        "acs_pct_graduate",
    ),
    # Walk + bike / workers total
    (
        ["B08301_019E", "B08301_018E"],
        ["B08301_019M", "B08301_018M"],
        "B08301_001E", "B08301_001M",
        "acs_pct_walk_bike",
    ),
    # No vehicle = owner_no_veh + renter_no_veh / total HH
    (
        ["B25044_003E", "B25044_010E"],
        ["B25044_003M", "B25044_010M"],
        "B25044_001E", "B25044_001M",
        "acs_pct_no_vehicle",
    ),
    # No insurance = 3 age groups / civ noninst pop
    (
        ["B27010_017E", "B27010_033E", "B27010_050E"],
        ["B27010_017M", "B27010_033M", "B27010_050M"],
        "B27010_001E", "B27010_001M",
        "acs_pct_no_insurance",
    ),
]


def get_all_vars() -> List[str]:
    """Get all unique Census variable codes (E and M) needed."""
    vars_set = set()
    for e, m, _ in DIRECT_VARS:
        vars_set.update([e, m])
    for ne, nm, de, dm, _ in RATIO_VARS:
        vars_set.update([ne, nm, de, dm])
    for ne_list, nm_list, de, dm, _ in COMPOSITE_VARS:
        vars_set.update(ne_list)
        vars_set.update(nm_list)
        vars_set.update([de, dm])
    return sorted(vars_set)


def fetch_zcta_data(api_key: Optional[str], variables: List[str]) -> pd.DataFrame:
    """
    Fetch ZCTA-level data directly from Census API.

    Census ACS5 supports `for=zip code tabulation area:*` — no crosswalk needed.
    """
    chunk_size = 48  # Census limit
    all_data = None

    for i in range(0, len(variables), chunk_size):
        chunk_vars = variables[i:i + chunk_size]
        var_str = ",".join(["NAME"] + chunk_vars)

        params = {
            "get": var_str,
            "for": "zip code tabulation area:*",
        }
        if api_key:
            params["key"] = api_key

        url = f"{ACS_API_BASE}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        log.info("Fetching chunk %d/%d (%d vars)...",
                 i // chunk_size + 1,
                 (len(variables) + chunk_size - 1) // chunk_size,
                 len(chunk_vars))

        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        headers = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=headers)

        if all_data is None:
            all_data = df
        else:
            merge_cols = [c for c in ["NAME", "zip code tabulation area"] if c in all_data.columns and c in df.columns]
            all_data = all_data.merge(df, on=merge_cols, how="outer")

        time.sleep(0.5)

    log.info("Fetched %d ZCTAs", len(all_data) if all_data is not None else 0)
    return all_data if all_data is not None else pd.DataFrame()


def propagate_ratio_moe(
    num_e: pd.Series,
    num_m: pd.Series,
    denom_e: pd.Series,
    denom_m: pd.Series,
) -> pd.Series:
    """
    Propagate MOE for a derived proportion using Census Bureau formula.

    MOE_pct = (1/denom) * sqrt(num_m^2 - (proportion^2 * denom_m^2))
    When inner term < 0:
    MOE_pct = (1/denom) * sqrt(num_m^2 + (proportion^2 * denom_m^2))

    Result is in percentage points (multiply proportion by 100).
    """
    proportion = num_e / denom_e.replace(0, np.nan)
    inner = num_m**2 - (proportion**2 * denom_m**2)

    # Use additive formula when subtractive yields negative
    inner_safe = np.where(inner >= 0, inner, num_m**2 + (proportion**2 * denom_m**2))

    moe_proportion = np.sqrt(inner_safe) / denom_e.replace(0, np.nan)
    # Convert to percentage points
    return moe_proportion * 100


def propagate_sum_moe(moe_list: List[pd.Series]) -> pd.Series:
    """MOE for sum of independent estimates: sqrt(sum(MOE_i^2))."""
    squared = [m**2 for m in moe_list]
    return np.sqrt(sum(squared))


def build_moe_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute MOE for each ACS feature."""
    # Standardize
    zcta_col = "zip code tabulation area"
    if zcta_col not in raw.columns:
        raise ValueError(f"Missing ZCTA column. Columns: {raw.columns.tolist()}")

    result = pd.DataFrame()
    result["zcta_id"] = raw[zcta_col].astype(str).str.zfill(5)

    # Convert all B-columns to numeric, replace Census sentinel values with NaN
    # Census uses -666666666 (estimate suppressed), -222222222 (MOE suppressed),
    # -999999999 (not available), etc.
    for col in raw.columns:
        if col.startswith("B"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw.loc[raw[col] < -1e6, col] = np.nan

    # --- Direct MOEs (just copy M column, scale to same units) ---
    for e_var, m_var, feature_name in DIRECT_VARS:
        if m_var in raw.columns:
            result[f"{feature_name}_moe"] = raw[m_var].astype(float)
        else:
            log.warning("Missing: %s", m_var)
            result[f"{feature_name}_moe"] = np.nan

    # --- Ratio MOEs (propagate) ---
    for num_e, num_m, denom_e, denom_m, feature_name in RATIO_VARS:
        if all(c in raw.columns for c in [num_e, num_m, denom_e, denom_m]):
            result[f"{feature_name}_moe"] = propagate_ratio_moe(
                raw[num_e].astype(float),
                raw[num_m].astype(float),
                raw[denom_e].astype(float),
                raw[denom_m].astype(float),
            )
        else:
            missing = [c for c in [num_e, num_m, denom_e, denom_m] if c not in raw.columns]
            log.warning("Missing for %s: %s", feature_name, missing)
            result[f"{feature_name}_moe"] = np.nan

    # --- Composite MOEs (sum then ratio) ---
    for num_e_list, num_m_list, denom_e, denom_m, feature_name in COMPOSITE_VARS:
        all_present = (
            all(c in raw.columns for c in num_e_list) and
            all(c in raw.columns for c in num_m_list) and
            denom_e in raw.columns and
            denom_m in raw.columns
        )
        if all_present:
            # Sum of numerators
            sum_e = sum(raw[c].astype(float).fillna(0) for c in num_e_list)
            # MOE of summed numerator
            sum_m = propagate_sum_moe([raw[c].astype(float).fillna(0) for c in num_m_list])
            # Propagate ratio
            result[f"{feature_name}_moe"] = propagate_ratio_moe(
                sum_e, sum_m,
                raw[denom_e].astype(float),
                raw[denom_m].astype(float),
            )
        else:
            log.warning("Missing composite vars for %s", feature_name)
            result[f"{feature_name}_moe"] = np.nan

    # Coverage flag: True if all MOE columns are non-null
    moe_cols = [c for c in result.columns if c.endswith("_moe")]
    result["has_acs_moe"] = result[moe_cols].notna().all(axis=1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch ACS MOE at ZCTA level")
    parser.add_argument("--output", default="/tmp/zcta_acs_margins_of_error.parquet")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to S3 after building")
    args = parser.parse_args()

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        log.warning("No CENSUS_API_KEY set. API may be rate-limited or fail.")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Get all variables needed
    all_vars = get_all_vars()
    log.info("Need %d Census variables (E + M)", len(all_vars))

    # Fetch ZCTA-level data directly
    raw = fetch_zcta_data(api_key, all_vars)
    if raw.empty:
        log.error("No data fetched. Check API key and network.")
        sys.exit(1)

    # Build MOE dataframe
    moe_df = build_moe_dataframe(raw)

    # Validate
    assert moe_df["zcta_id"].is_unique, "Duplicate ZCTA IDs"
    moe_cols = [c for c in moe_df.columns if c.endswith("_moe")]

    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:      %d", len(moe_df))
    log.info("MOE cols:   %d", len(moe_cols))
    log.info("Coverage:   %d / %d have full MOEs",
             moe_df["has_acs_moe"].sum(), len(moe_df))

    # Sample MOE values
    log.info("")
    log.info("=== SAMPLE MOEs ===")
    for col in ["acs_median_hh_income_moe", "acs_pct_below_poverty_moe",
                "acs_total_pop_moe"]:
        if col in moe_df.columns:
            s = moe_df[col]
            log.info("  %s: mean=%.2f, median=%.2f, max=%.2f",
                     col, s.mean(), s.median(), s.max())

    # Save
    moe_df.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/zcta_acs_margins_of_error.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "fetch_acs_moe",
            "timestamp": timestamp,
            "source": ACS_API_BASE,
            "vintage": "2022",
            "product": "acs5",
            "resolution": "zcta",
            "n_zctas": len(moe_df),
            "n_moe_columns": len(moe_cols),
            "moe_columns": moe_cols,
            "coverage_has_moe": int(moe_df["has_acs_moe"].sum()),
            "propagation_method": "census_bureau_ratio_formula",
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/zcta_acs_moe_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
