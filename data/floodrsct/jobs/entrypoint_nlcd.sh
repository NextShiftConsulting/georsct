#!/bin/bash
set -e
pip install -q requests boto3
python3 -u /opt/ml/processing/input/code/fetch_nlcd_impervious.py
