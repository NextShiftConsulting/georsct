#!/bin/bash
set -e
pip install -q scikit-image \
    /opt/ml/processing/input/wheels/yrsn-*.whl \
    /opt/ml/processing/input/wheels/yrsn_controlplane-*.whl
export PYTHONPATH=/opt/ml/processing/input/code:$PYTHONPATH
python3 -u /opt/ml/processing/input/code/run_s019a.py "$@"
