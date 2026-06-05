# DOE: S018B-FREEZE — GeoRSCT-Bench Benchmark Freeze

**Experiment ID:** S018B-FREEZE
**Namespace:** GeoRSCT-Bench
**Claim:** C4 (GeoRSCT-Bench is a usable ZCTA benchmark for geospatial solver compatibility)
**Status:** DRAFT
**Created:** 2026-05-02
**Registry ref:** EXPERIMENT_REGISTRY.md, Day 2

---

## Abstract

Freeze all benchmark release artifacts for the NeurIPS 2026 submission. This is an engineering
experiment: no hypothesis testing, but strict acceptance criteria that the frozen artifacts are
complete, internally consistent, reproducible from a single command, and free of spatial leakage.

---

## Deliverables

Each deliverable has a file name per the ADR-018 artifact naming convention and an acceptance
criterion that must be verified before the artifact is locked.

### D1: Benchmark Manifest

**File:** `bench_manifest_v1.json`

Contents:
- 27 task names, families (health / socioeconomic / environmental), CDC PLACES or ACS source
- 33 ACS feature names with Census table IDs
- Canonical ZCTA count (31,789) and SHA256 hash
- ACS vintage (2022 5-Year, released 2023-12-07)
- Race/ethnicity standard note (1997 OMB SPD 15)
- Solver family descriptions (PCA32-GBDT, Spatial-Lag-GBDT, GNN GraphSAGE)
- OOF artifact provenance (S3 keys, git hashes from provenance.json)
- Version identifier

**Acceptance:** JSON schema validates; ZCTA count matches DATA_MANIFEST.md; SHA256 matches;
all 27 tasks present; all 33 features present.

### D2: Frozen Splits

**File:** `bench_splits_v1.json`

Contents:
- Interpolation split (county holdout, 364/1823 counties, 24864 train / 6925 test)
- Extrapolation split (state holdout, 9 states, 26586 train / 5203 test)
- Split provenance (seed, method, date)
- Adjacency-isolated ZCTA list (47 ZCTAs)

**Acceptance:** Counts match DATA_MANIFEST.md; no train/test ZCTA overlap; no county/state
leaks across boundary; intersection with canonical set is exact.

### D3: Spatial Leakage Audit

**File:** `bench_leakage_audit_v1.json`

Checks:
- No target-lag columns in feature set (lag_target_* must not exist)
- County holdout: no county appears in both train and test
- State holdout: no state appears in both train and test
- Moran's I on residuals by solver family (informational, not a gate)
- Adjacency edge audit: no train-test edges cross the split boundary
  at the county level (interpolation) or state level (extrapolation)

**Acceptance:** All leakage checks pass. Moran's I reported but not gated.

### D4: Solver Leaderboard

**File:** `bench_scorecard_s018.parquet`

Contents per row:
- task, solver_family, split (interpolation/extrapolation)
- R2_test, R2_train, TRF, spread (max-min R2 across families)
- OOF artifact S3 key

**Acceptance:** 27 tasks x 3 families x 2 splits = 162 rows (or 81 if interpolation-only
for v1). All R2 values match canonical OOF parquets. Spread matches Table 2.

### D5: Scorecard Summary

**File:** `bench_scorecard_summary_v1.md`

Human-readable summary:
- Per-task ranking table sorted by N-ceiling
- Cross-family spread statistics (mean 0.037, median 0.030)
- Architecture-invariance statement (with EVIDENCE_AUDIT-corrected numbers)
- Known limitations (3 families only, ACS features only, CONUS-only)

**Acceptance:** Numbers match D4. No stale 0.02 or factor-of-20 values.

### D6: One-Command Reproducibility

**File:** `reproduce.sh`

Script that:
1. Downloads canonical OOF artifacts from S3
2. Loads frozen splits
3. Computes per-task R2 by solver family
4. Produces scorecard parquet
5. Runs leakage audit
6. Compares against frozen manifest

**Acceptance:** Clean checkout + `bash reproduce.sh` produces identical scorecard.
Deterministic under fixed seeds. No network dependency beyond S3.

---

## Input Artifacts (Already Exist)

| Artifact | Location | Status |
|----------|----------|--------|
| zcta_features_labels.parquet | S3: series_018/processed/ | Frozen |
| train_test_split.json | S3: series_018/processed/ | Frozen |
| oof_pca_v1.parquet | S3: series_018/oof_artifacts/ | Frozen |
| oof_spatial_lag_v1.parquet | S3: series_018/oof_artifacts/ | Frozen |
| oof_gnn_v2.parquet | S3: series_018/oof_artifacts/ | Frozen |
| provenance.json | S3: series_018/oof_artifacts/ | Frozen |
| DATA_MANIFEST.md | yrsn-experiments/exp/series_018/ | Reference |

---

## Execution Plan

This experiment runs locally (no SageMaker). All inputs are on S3; outputs are JSON/parquet
files committed to the paper repo.

1. Download canonical artifacts from S3 to local cache
2. Generate manifest (D1) from DATA_MANIFEST.md + provenance.json
3. Validate splits (D2) against downloaded data
4. Run leakage audit (D3) on splits + features
5. Compute scorecard (D4) from OOF parquets
6. Generate summary (D5)
7. Write reproduce.sh (D6) and verify it works

---

## Success Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| All 6 deliverables present | 6/6 | PENDING |
| Leakage audit clean | 0 violations | PENDING |
| Scorecard matches Table 2 | mean spread 0.037, N-ceiling 0.155-0.593 | PENDING |
| reproduce.sh runs clean | exit code 0 on fresh checkout | PENDING |
| No stale audit values | 0 matches for 0.02/factor-20 | PENDING |

---

## Kill Condition

None. This experiment is required for submission. If blocked, the deadline slips.

---

## Change Control

| Date | Change | Reason |
|------|--------|--------|
| 2026-05-02 | Initial draft | Day 1 scaffold |
