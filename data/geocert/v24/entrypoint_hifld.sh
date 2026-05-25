#!/bin/bash
set -euo pipefail

echo "=== HIFLD Hospital & Pharmacy Feature Build ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"

# Install dependencies not in base pytorch image
pip install --quiet pyarrow

# Code is mounted at /opt/ml/processing/input/code
# HIFLD CSV is mounted at /opt/ml/processing/input/hifld/
# ZCTA features are mounted at /opt/ml/processing/input/data/

cd /opt/ml/processing/input/code

python3 -u build_hifld_features.py \
    --zcta-data /opt/ml/processing/input/data/zcta_features_labels.parquet \
    --output /opt/ml/processing/output/hifld_zcta.parquet \
    --upload

echo "=== DONE ==="
