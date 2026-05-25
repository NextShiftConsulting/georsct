#!/bin/bash
set -e

# No geo dependencies needed — pure requests + pandas
pip install -q pyarrow

python3 -u /opt/ml/processing/input/code/run_nfip.py \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output \
    "$@"
