#!/bin/bash
set -e
pip install -q requests boto3 pandas pyarrow numpy swarm-auth
python3 -u /opt/ml/processing/input/code/fetch_noaa_slosh.py
