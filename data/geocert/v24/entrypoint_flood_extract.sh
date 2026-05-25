#!/bin/bash
set -e

pip install -q fiona

python3 -u /opt/ml/processing/input/code/run_flood_extract.py \
    --output-dir /opt/ml/processing/output
