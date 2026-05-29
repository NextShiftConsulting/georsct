#!/bin/bash
set -e
pip install -q requests boto3 pandas pyarrow geopandas swarm-auth
python3 -u /opt/ml/processing/input/code/fetch_nyc_sewersheds.py
