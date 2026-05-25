#!/bin/bash
set -e

pip install -q geopandas pyogrio shapely pyarrow requests

python3 -u /opt/ml/processing/input/code/run_flood_fetch_only.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --crosswalk /opt/ml/processing/input/data/zcta_county_crosswalk.parquet \
    --output-dir /opt/ml/processing/output \
    --threads 16
