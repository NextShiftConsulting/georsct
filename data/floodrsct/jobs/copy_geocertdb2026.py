#!/usr/bin/env python3
"""
copy_geocertdb2026.py -- SageMaker job: copy reusable ZCTA features from
geocertdb2026 into the s035 dedicated bucket.

Source: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/
Target: s3://swarm-floodrsct-data/raw/geocertdb2026/

No reprocessing — this is a pure S3-to-S3 copy. The container needs IAM
access to both buckets (SageMakerExecutionRole covers both swarm-* buckets).

Also filters to Houston, New Orleans, and NYC ZCTAs and writes scenario
subsets to s3://swarm-floodrsct-data/raw/geocertdb2026/scenarios/.
"""

import logging
import sys
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

SRC_BUCKET = "swarm-yrsn-datasets"
SRC_PREFIX = "rsct_curriculum/series_018/processed"
DST_BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/geocertdb2026"

# Files to copy
PARQUET_FILES = [
    "zcta_features_labels.parquet",
    "svi_zcta.parquet",
    "flood_zones_zcta.parquet",
    "noaa_storm_events_zcta.parquet",
    "nfip_claims_zcta.parquet",
    "twi_features_zcta.parquet",
    "zcta_county_crosswalk.parquet",
]

# ZCTA state prefixes for scenario filtering
# ZCTAs starting with these prefixes are in the target states
SCENARIO_ZCTA_STATES = {
    "houston": {"state": "TX", "county_fips": "48201"},
    "new_orleans": {"state": "LA", "county_fips": "22071"},
    "nyc": {"state": "NY", "county_fips_list": ["36061", "36047", "36081", "36005", "36085"]},
    "riverside_coachella": {"state": "CA", "county_fips_list": ["06065", "06025"]},
    "southwest_florida": {
        "state": "FL",
        "county_fips_list": ["12021", "12071", "12115", "12081", "12057", "12103"],
    },
}


def s3_copy(s3, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
    )
    log.info("Copied s3://%s/%s -> s3://%s/%s", src_bucket, src_key, dst_bucket, dst_key)


def filter_and_upload(
    s3, df: pd.DataFrame, county_col: str, county_values: list[str],
    scenario: str, filename: str
) -> None:
    """Filter a ZCTA parquet to a scenario's counties and re-upload."""
    if county_col not in df.columns:
        log.warning("Column %s not in %s; skipping scenario filter", county_col, filename)
        return

    mask = df[county_col].isin(county_values)
    filtered = df[mask].copy()
    log.info("Scenario %s: %d / %d ZCTAs in %s", scenario, len(filtered), len(df), filename)

    local_path = f"/tmp/{scenario}_{filename}"
    filtered.to_parquet(local_path, index=False)
    dst_key = f"{DST_PREFIX}/scenarios/{scenario}/{filename}"
    s3.upload_file(local_path, DST_BUCKET, dst_key)
    log.info("Uploaded scenario subset to s3://%s/%s", DST_BUCKET, dst_key)


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    # Step 1: copy all files wholesale
    for filename in PARQUET_FILES:
        src_key = f"{SRC_PREFIX}/{filename}"
        dst_key = f"{DST_PREFIX}/{filename}"
        try:
            s3_copy(s3, SRC_BUCKET, src_key, DST_BUCKET, dst_key)
        except s3.exceptions.ClientError as e:
            if "NoSuchKey" in str(e):
                log.warning("Source not found: s3://%s/%s — skipping", SRC_BUCKET, src_key)
            else:
                raise

    # Step 2: load crosswalk and main features for scenario subsetting
    log.info("Loading crosswalk for scenario subsetting...")
    xwalk_local = "/tmp/zcta_county_crosswalk.parquet"
    s3.download_file(DST_BUCKET, f"{DST_PREFIX}/zcta_county_crosswalk.parquet", xwalk_local)
    xwalk = pd.read_parquet(xwalk_local)

    # Identify county FIPS column
    county_col = next(
        (c for c in xwalk.columns if "county" in c.lower() and "fips" in c.lower()),
        next((c for c in xwalk.columns if "county" in c.lower()), None)
    )
    log.info("Crosswalk county column: %s", county_col)

    if county_col is None:
        log.warning("Cannot identify county column in crosswalk; skipping scenario subsets")
        return

    for scenario, spec in SCENARIO_ZCTA_STATES.items():
        county_values = spec.get("county_fips_list", [spec.get("county_fips")])
        county_values = [v for v in county_values if v]

        # Identify ZCTAs in this scenario
        if "zcta_id" not in xwalk.columns and "zcta5" not in xwalk.columns:
            log.warning("No zcta_id column in crosswalk; skipping %s", scenario)
            continue

        zcta_col = "zcta_id" if "zcta_id" in xwalk.columns else "zcta5"
        scenario_zctas = xwalk[xwalk[county_col].isin(county_values)][zcta_col].unique()
        log.info("Scenario %s: %d unique ZCTAs from crosswalk", scenario, len(scenario_zctas))

        # Filter the main features file
        main_local = "/tmp/zcta_features_labels.parquet"
        s3.download_file(DST_BUCKET, f"{DST_PREFIX}/zcta_features_labels.parquet", main_local)
        main_df = pd.read_parquet(main_local)

        main_zcta_col = next(
            (c for c in main_df.columns if "zcta" in c.lower()),
            None
        )
        if main_zcta_col is None:
            log.warning("No zcta column in main features; skipping %s", scenario)
            continue

        filtered = main_df[main_df[main_zcta_col].isin(scenario_zctas)].copy()
        local_path = f"/tmp/{scenario}_zcta_features.parquet"
        filtered.to_parquet(local_path, index=False)
        s3_key = f"{DST_PREFIX}/scenarios/{scenario}/zcta_features_labels.parquet"
        s3.upload_file(local_path, DST_BUCKET, s3_key)
        log.info("Uploaded s3://%s/%s (%d rows)", DST_BUCKET, s3_key, len(filtered))

    log.info("copy_geocertdb2026 complete")


if __name__ == "__main__":
    main()
