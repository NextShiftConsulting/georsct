#!/usr/bin/env python3
"""
clean_acs_negatives.py -- Fix residual bogus negative values in ACS feature columns.

fix_acs_sentinels.py caught the Census Bureau sentinel (-666666666), but 6 ACS
median-value columns also contain large negative floats (~-700K to -999K) that
survived. These are artifacts from upstream preprocessing, not real values.

Affected columns (825 cells across 387 ZCTAs):
  acs_gini_index         132 negative values
  acs_median_age          71 negative values
  acs_median_hh_income   130 negative values
  acs_median_home_value   94 negative values
  acs_median_rent        250 negative values
  acs_median_year_built  148 negative values

Action: Replace all negative values in these columns with NaN.

Usage:
    python clean_acs_negatives.py --input geocert_table.parquet --output geocert_table_clean.parquet
    python clean_acs_negatives.py --input geocert_simplified_001.geoparquet --output geocert_simplified_001_clean.geoparquet --geo
"""

import argparse
import logging
import sys
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

# Columns with known bogus negative values
AFFECTED_COLUMNS = [
    "acs_gini_index",
    "acs_median_age",
    "acs_median_hh_income",
    "acs_median_home_value",
    "acs_median_rent",
    "acs_median_year_built",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--geo", action="store_true", help="Input is GeoParquet")
    args = parser.parse_args()

    src = Path(args.input)
    if args.geo:
        import geopandas as gpd
        df = gpd.read_parquet(src)
    else:
        df = pd.read_parquet(src)

    log.info("Loaded %d rows from %s", len(df), src)

    total_fixed = 0
    for col in AFFECTED_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col] < 0
        n_neg = int(mask.sum())
        if n_neg > 0:
            df.loc[mask, col] = np.nan
            total_fixed += n_neg
            log.info("  %s: %d negative -> NaN", col, n_neg)

    log.info("Total cells fixed: %d", total_fixed)

    out = Path(args.output)
    df.to_parquet(out, index=False)
    log.info("Written: %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
