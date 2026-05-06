#!/bin/bash
# bootstrap_s019d.sh -- S019D bootstrap CI container entrypoint
set -e
export PYTHONPATH=/opt/ml/processing/input/code:$PYTHONPATH
python3 -u /opt/ml/processing/input/code/run_s019d_bootstrap.py "$@"
