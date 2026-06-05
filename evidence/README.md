# Evidence Artifacts for GeoRSCT (NeurIPS 2026)

All artifacts referenced by `V3/combine_georsct.tex`, collected from
`georsct/`, `yrsn-train/`, and `rsct-geocert/exp/` for single-repo traceability.

## Layout

```
evidence/
  specifications/       # Taxonomy, injection results, Croissant, manifest
  diagnostics/          # Injection harness, task residual floor, certificate pipeline, OOF generators
  solvers/              # PCA32-GBDT, Spatial Lag-GBDT, GraphSAGE GNN training code
  certificates/         # Issued certificate triples (alpha, kappa, sigma)
  predictions/          # Solver metrics, distances, clustering, PCA loadings
  experiments/          # SageMaker launchers and configs (S018B, S018D, S018H)
  validation/           # Spatial representation QA (9-check module)
```

## Paper Section -> Evidence Map

| Paper Section | Claim | Evidence File(s) |
|---|---|---|
| S3 (Certificate Triple) | alpha = R/(R+N), kappa = R*(1-N), sigma = std(kappa_i) | `diagnostics/certificate_issuer.py` |
| S3 (Audit Pipeline) | 5-gate sequential evaluator, oobleck threshold | `diagnostics/certificate_audit.py` |
| S4 (Taxonomy) | 6 modes, 3 categories, signal patterns | `specifications/georsct_taxonomy.json` |
| S5 (Injection) | 9/9 top-1 classification, 8 seeds | `specifications/injection_validation.json`, `diagnostics/stress_geocert.py` |
| S6 (task residual floor) | TRF = 1 - max_k(R^2_k), block bootstrap B=1000 | `diagnostics/task_residual_floor_estimator.py` |
| S7 (Results) | Table 2 (CONUS-27 R^2 by family) | `predictions/solver_metrics.parquet`, `predictions/target_family_metrics.parquet` |
| S7 (Results) | Certificate triples per solver-task pair | `certificates/certificate_rsn.parquet`, `predictions/certificate_rsn.parquet` |
| S7 (Results) | Cross-family spread, solver distances | `predictions/pairwise_solver_distances.csv`, `predictions/solver_metric_summary.csv` |
| S7 (Figures) | Fig 4 heatmap, Fig 5 spread vs N-ceil | `../figures/fig4_nceiling_heatmap.py`, `../figures/fig5_spread_vs_nceiling.py` |
| App B (Benchmark) | 31,789 ZCTAs, 33 ACS features, 27 targets | `../data/geocert/v24/zcta_features_labels.parquet` |
| App B (Splits) | County-holdout, state-holdout protocols | `../data/geocert/v24/geocert_splits_v24.parquet` |
| App B (Enrichment) | SVI, HIFLD, drive times, flood, CDC CI, ACS MOE | `../data/geocert/v24/*.parquet` |
| App C (Solvers) | PCA32-GBDT, Spatial Lag-GBDT | `solvers/train_and_export_v2.py` |
| App C (Solvers) | GraphSAGE GNN + GBDT | `solvers/train_and_export_gnn_v2.py` |
| App D (task residual floor) | Synthetic validation, bootstrap CIs | `diagnostics/task_residual_floor_estimator.py`, `../figures/fig8_*.py` |
| App F (Experiments) | S018B freeze, S018D sweep, S018H canonical | `experiments/run_s018b_freeze.py`, `experiments/run_s018d_solver_diversity.py`, `experiments/sagemaker_s018h.py` |
| App G (Failure Modes) | Per-mode exemplars and diagnostics | `diagnostics/certificate_mixture.py`, `specifications/georsct_taxonomy.json` |
| App H (Reproducibility) | Spatial validation, leakage checks | `validation/spatial.py` |
| Data Statement | Croissant metadata for HF dataset | `specifications/croissant.json` |

## Canonical Data Sources (in-repo, not copied here)

| Artifact | Path |
|---|---|
| CONUS-27 numeric values (Table 2 source) | `V2/figures/figures.py` line 95 (CONUS_27 array) |
| Figure generation scripts | `figures/fig*.py` |
| Benchmark build pipeline (11 scripts) | `data/geocert/v24/build_*.py`, `fetch_*.py`, `run_*.py` |
| All 8 data parquets | `data/geocert/v24/*.parquet` |

## Known Gaps

1. **Table 2 XXX placeholders** - Interpolation values computable from `V2/figures/figures.py` CONUS_27.
   Extrapolation values (S018B) not yet generated.
2. **Appendix F experiment ledger** - S018A/B/E/H results columns are XXX.
3. **Raw per-ZCTA OOF predictions** - Aggregates exist; per-sample outputs on S3 (not local).
