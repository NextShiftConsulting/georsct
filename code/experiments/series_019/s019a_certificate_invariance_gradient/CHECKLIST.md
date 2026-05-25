# S019A Pre-Flight Checklist

## Git State
- [ ] `git status` clean (no uncommitted changes)
- [ ] `git push` complete
- [ ] Commit hash recorded: _______________

## AWS Infrastructure
- [ ] S3 data exists: `swarm-yrsn-datasets/rsct_curriculum/series_018/processed/zcta_features_labels.parquet`
- [ ] S3 representations exist: `swarm-yrsn-datasets/rsct_curriculum/series_018/artifacts/representations/`
- [ ] Wheel uploaded: `swarm-yrsn-datasets/rsct_code/wheels/` (with compute_sigma_request)
- [ ] Output prefix clear: `swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019a/`

## Dry Run
- [ ] `python run_s019a.py --data-dir /tmp/data --repr-dir /tmp/repr --output-dir /tmp/out --dry-run` passes
- [ ] Reports 27 cells, 135 fits

## Launch
- [ ] Instance: ml.m5.xlarge (4 vCPU, 16 GB)
- [ ] Estimated runtime: < 30 min
- [ ] No duplicate job running

## Execution Record

| Field | Value |
|-------|-------|
| Job name | |
| Commit hash | |
| Launch time | |
| Instance | |
| Wall-clock | |
| Status | |
