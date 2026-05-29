#!/bin/bash
set -e
pip install -q boto3 pandas pyarrow swarm-auth
python3 -u /opt/ml/processing/input/code/copy_geocertdb2026.py
