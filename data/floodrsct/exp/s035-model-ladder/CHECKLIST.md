# S035-Model-Ladder Pre-Flight Checklist

**Bucket:** `s3://swarm-floodrsct-data/`
**Job scripts:** `data/floodrsct/jobs/`
**Launchers:** `data/floodrsct/scripts/`

---

## 1. Git State

- [ ] `git status` -- working tree clean
- [ ] `git log -1 --oneline` -- commit hash: `__________`
- [ ] `git push`

---

## 2. Data Prerequisites

### Assembled Parquets

| Scenario | S3 Key | Status |
|----------|--------|--------|
| houston | `processed/houston/houston_event_features.parquet` | [ ] |
| new_orleans | `processed/new_orleans/no_event_features.parquet` | [x] |
| southwest_florida | `processed/southwest_florida/swfl_event_features.parquet` | [x] |
| nyc | `processed/nyc/nyc_event_features.parquet` | [ ] |
| riverside_coachella | `processed/riverside_coachella/rc_event_features.parquet` | [ ] |

### Reference Data

- [ ] `raw/geocertdb2026/zcta_county_crosswalk.parquet`
- [ ] `raw/geocertdb2026/zcta_adjacency.parquet` (optional, folds fall back to ZIP3)

---

## 3. Phase 0: Audit Battery (existing orchestrator)

```bash
python scripts/launch_strat_sampler_qa.py --scenario houston
# OR run stratified_coverage_audit.py directly on SageMaker
```

- [ ] All P1-P6 support probes PASS
- [ ] Results at `evidence/qa/coverage_audit_{scenario}.json`

---

## 4. Phase 1: R0 Baseline (folds + training in one job)

### Dry Run

```bash
python scripts/launch_train_r0_baseline.py --scenario southwest_florida --dry-run
```

- [ ] Dry run shows ml.m5.xlarge, 10 GB volume, correct image

### Launch

```bash
python scripts/launch_train_r0_baseline.py --scenario southwest_florida
```

- [ ] Job starts successfully

### Post-Run Validation

- [ ] `folds/{scenario}_folds.parquet` on S3 with 3 fold columns, no NaN
- [ ] `folds/{scenario}_folds_meta.json` shows block_strategy (county or zip3)
- [ ] `results/s035/r0_{scenario}.json` on S3
- [ ] At least 1 target trained (obs_nfip_event_claims always available)
- [ ] RMSE/baseline < 1.0 for HistGBDT on at least one target (H1 check)

---

## 5. Execution Record

| Phase | Date | Commit | Job Name | Duration | Result |
|-------|------|--------|----------|----------|--------|
| 0 (audit) | | | | | |
| 1 (R0) | | | | | |
| 2 (R1) | | | | | |
| 3 (R2) | | | | | |
| 4 (uplift) | | | | | |
