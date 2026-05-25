#!/bin/bash
set -e

pip install -q requests

python3 -u /opt/ml/processing/input/code/run_flood_bulk_download.py \
    --catalog /opt/ml/processing/input/code/nfhl_download_catalog.json \
    --output-dir /opt/ml/processing/output \
    --threads 16
