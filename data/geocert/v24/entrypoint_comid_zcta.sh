#!/bin/bash
set -e

# Spatial + 7zip dependencies
pip install -q geopandas pyogrio shapely pyarrow py7zr

VPU_ARGS=""
if [ -n "${NHDPLUS_VPUS}" ]; then
    VPU_ARGS="--vpus ${NHDPLUS_VPUS}"
fi

python3 -u /opt/ml/processing/input/code/run_comid_zcta.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --output-dir /opt/ml/processing/output \
    --cache-dir /tmp/nhdplus_cache \
    ${VPU_ARGS}
