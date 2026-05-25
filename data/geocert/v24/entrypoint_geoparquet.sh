#!/bin/bash
set -euo pipefail

echo "=== GeoParquet Assembly Build ==="
echo "Date: $(date -u)"
echo "Python: $(python3 --version)"

# Install dependencies not in base pytorch image
pip install --quiet geopandas pyogrio pyarrow requests huggingface_hub

# Fetch HF token from Secrets Manager
export HF_TOKEN=$(python3 -c "
import boto3, json
sm = boto3.client('secretsmanager', region_name='us-east-1')
s = sm.get_secret_value(SecretId='prod/yrsn/hf_token')
v = s['SecretString']
try:
    v = json.loads(v)
    v = v.get('HF_TOKEN', v.get('token', list(v.values())[0]))
except Exception:
    pass
print(v, end='')
" 2>/dev/null) || echo "WARNING: Could not fetch HF_TOKEN"
echo "HF_TOKEN set: $([ -n \"$HF_TOKEN\" ] && echo yes || echo no)"

# Code is mounted at /opt/ml/processing/input/code
cd /opt/ml/processing/input/code

python3 -u build_geoparquet.py \
    --simplify 0.001 \
    --output /opt/ml/processing/output/georsct_simplified_001.geoparquet

echo "=== DONE ==="
