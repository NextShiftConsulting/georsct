#!/bin/bash
set -e

pip install -q geopandas pyogrio shapely pyarrow

python3 -u /opt/ml/processing/input/code/run_flood_overlay_only.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output
