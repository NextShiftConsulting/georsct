#!/bin/bash
set -e

# Install geo dependencies not in the PyTorch base image
pip install -q geopandas pyogrio shapely pyarrow

python3 -u /opt/ml/processing/input/code/run_flood_zones.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output \
    --threads 8
