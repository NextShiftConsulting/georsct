#!/bin/bash
set -e

# No geo dependencies needed — pure requests + pandas
pip install -q pyarrow

EXTRA_ARGS=""
if [ -n "${NFIP_MAX_PAGES}" ]; then
    EXTRA_ARGS="--max-pages ${NFIP_MAX_PAGES}"
fi

python3 -u /opt/ml/processing/input/code/run_nfip.py \
    --data-dir /opt/ml/processing/input/data \
    --output-dir /opt/ml/processing/output \
    ${EXTRA_ARGS}
