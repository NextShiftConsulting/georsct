#!/bin/bash
set -e
pip install -q boto3 pandas pyarrow /opt/ml/processing/input/code/swarm_auth-0.2.0-py3-none-any.whl
python3 -u /opt/ml/processing/input/code/copy_geocertdb2026.py
