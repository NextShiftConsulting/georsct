#!/bin/bash
set -e

# Spatial dependencies (pynhd for WaterData WFS; py7zr no longer needed)
pip install -q geopandas pyogrio shapely pyarrow pynhd

HUC2_ARGS=""
if [ -n "${NHDPLUS_VPUS}" ]; then
    HUC2_ARGS="--huc2 ${NHDPLUS_VPUS}"
fi

python3 -u /opt/ml/processing/input/code/run_comid_zcta.py \
    --tiger-dir /opt/ml/processing/input/tiger \
    --output-dir /opt/ml/processing/output \
    --cache-dir /tmp/nhdplus_cache \
    ${HUC2_ARGS}
