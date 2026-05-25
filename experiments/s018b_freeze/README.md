# S018B-FREEZE: GeoRSCT-Bench Benchmark Freeze

Freezes all benchmark release artifacts for the NeurIPS 2026 submission.

## Claim

C4: GeoRSCT-Bench is a usable ZCTA benchmark for geospatial solver compatibility.

## What This Produces

1. `bench_manifest_v1.json` — canonical task/feature/ZCTA manifest
2. `bench_splits_v1.json` — frozen train/test splits (interpolation + extrapolation)
3. `bench_leakage_audit_v1.json` — spatial leakage validation
4. `bench_scorecard_s018.parquet` — solver leaderboard by task
5. `bench_scorecard_summary_v1.md` — human-readable summary
6. `reproduce.sh` — one-command reproducibility script

## Running

```bash
# 1. Lock DOE (change Status to LOCKED)
# 2. Commit and push
# 3. Run freeze script
python freeze_benchmark.py

# 4. Verify
bash reproduce.sh
```

## Input Data

All inputs on S3 under `s3://swarm-yrsn-datasets/rsct_curriculum/series_018/`.
See DOE_LOCKED.md for full inventory.
