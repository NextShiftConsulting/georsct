# S018B-FREEZE Pre-Flight Checklist

## Git State

- [ ] All experiment files committed
- [ ] Pushed to remote
- [ ] Commit hash recorded: _______________

## Input Artifacts

- [ ] zcta_features_labels.parquet accessible on S3
- [ ] train_test_split.json accessible on S3
- [ ] oof_pca_v1.parquet accessible on S3
- [ ] oof_spatial_lag_v1.parquet accessible on S3
- [ ] oof_gnn_v2.parquet accessible on S3
- [ ] provenance.json accessible on S3
- [ ] DATA_MANIFEST.md read and cross-referenced

## Deliverable Validation

- [ ] D1: bench_manifest_v1.json — schema validates, counts match
- [ ] D2: bench_splits_v1.json — no train/test overlap, counts match
- [ ] D3: bench_leakage_audit_v1.json — all checks pass
- [ ] D4: bench_scorecard_s018.parquet — 27x3 rows, R2 matches OOF
- [ ] D5: bench_scorecard_summary_v1.md — no stale 0.02 or factor-20
- [ ] D6: reproduce.sh — exit code 0 on fresh checkout

## Cross-Check

- [ ] Mean spread matches Table 2 (0.037)
- [ ] N-ceiling range matches Table 2 (0.155-0.593)
- [ ] Family means match EVIDENCE_AUDIT (PCA 0.655, GNN 0.651, SLag 0.666)

## Execution Record

| Run | Date | Commit | Result |
|-----|------|--------|--------|
| | | | |
