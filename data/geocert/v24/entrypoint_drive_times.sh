#!/bin/bash
set -euo pipefail

echo "=== Drive-Time Feature Build ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"

# Install dependencies not in base pytorch image
pip install --quiet pyarrow

# Code is mounted at /opt/ml/processing/input/code
# HIFLD CSV is mounted at /opt/ml/processing/input/hifld/
# ZCTA data is mounted at /opt/ml/processing/input/data/

cd /opt/ml/processing/input/code

python3 -u build_drive_times.py \
    --zcta-data /opt/ml/processing/input/data/zcta_features_labels.parquet \
    --output /opt/ml/processing/output/drive_times_zcta.parquet \
    --upload

echo "=== DONE ==="
