#!/bin/bash
set -e
pip install -q requests boto3 swarm-auth
python3 -u /opt/ml/processing/input/code/fetch_noaa_mrms_v2.py --event "$1"
