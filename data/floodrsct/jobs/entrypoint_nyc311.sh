#!/bin/bash
set -e
pip install -q requests boto3 pandas pyarrow swarm-auth
python3 -u /opt/ml/processing/input/code/fetch_nyc_311.py
