#!/bin/bash
set -e

# No geo dependencies — pure requests + pandas
pip install -q pyarrow

python3 -u /opt/ml/processing/input/code/run_noaa.py \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output \
    --year-start 1996 \
    --year-end 2024 \
    "$@"
