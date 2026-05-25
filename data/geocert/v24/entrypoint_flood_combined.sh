#!/bin/bash
set -e

# PyTorch CPU image has Python 3.11, modern numpy/pandas
pip install -q geopandas fiona pyogrio shapely pyarrow

python3 -u /opt/ml/processing/input/code/run_flood_combined.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output
