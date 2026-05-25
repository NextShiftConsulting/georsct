#!/bin/bash
set -e

pip install -q requests pyarrow

python3 -u /opt/ml/processing/input/code/run_twi.py \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output
