#!/bin/bash
set -e

# PyTorch CPU image has Python 3.11, modern numpy/pandas
pip install -q geopandas fiona pyogrio shapely pyarrow

# Find the county list JSON (shard-specific, placed by SageMaker input)
COUNTY_LIST=$(find /opt/ml/processing/input/county_list -name "*.json" | head -1)
echo "County list: $COUNTY_LIST"

python3 -u /opt/ml/processing/input/code/run_flood_shard.py \
    --county-list "$COUNTY_LIST" \
    --tiger-dir /opt/ml/processing/input/tiger \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output
