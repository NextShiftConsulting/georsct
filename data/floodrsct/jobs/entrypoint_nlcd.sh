#!/bin/bash
set -e
pip install -q requests boto3 /opt/ml/processing/input/code/swarm_auth-0.2.0-py3-none-any.whl
python3 -u /opt/ml/processing/input/code/fetch_nlcd_impervious.py
