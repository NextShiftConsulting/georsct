# S018D Artifact Inventory

**Date**: 2026-05-01
**Scope**: All S018D-related files across rsct-geocert and yrsn-experiments
**Total files**: 31

---

## 1. What Exists (by location)

### A. `rsct-geocert/data/s018d/` -- Raw Experiment Output (4 files)

| File | Size | Granularity | Description |
|------|------|-------------|-------------|
| `certificate_rsn.parquet` | 48 KB | 324 rows (12 solvers x 27 targets) | Per-solver-per-target RSN certificate records |
| `solver_metrics.parquet` | 16 KB | 12 rows (per solver) | Aggregate: R, S, N, alpha, kappa, sigma, omega, tau, entropy, collapse_risk |
| `target_family_metrics.parquet` | 12 KB | Per target family | Metrics broken out by target family (health, socioeconomic, environmental) |
| `summary.json` | 8 KB | 1 record | Top-level: 12 solvers, 27 targets, 31,789 ZCTAs, SEPARATION_CONFIRMED (4/5 conditions) |

**Note**: These are exact copies of `yrsn-experiments/s018d_solver_diversity/results/`. No transformation applied.

### B. `rsct-geocert/exp/s018d_posthoc/` -- Post-Hoc Analysis (10 files)

| File | Size | Description |
|------|------|-------------|
| `analyze.py` | 16 KB | PCA, clustering, pairwise distance analysis on 12-solver certificate metrics |
| `config.json` | 4 KB | Experiment config (experiment_id=s018d_posthoc) |
| `README.md` | 4 KB | Overview: aggregate metrics have structure but miss noisy-solver pathology |
| `results/summary.json` | 4 KB | Status=PASS, 5 PCs explain 100% variance |
| `results/paper_note.md` | 4 KB | Recommended paper sentence + interpretation |
| `results/solver_metric_summary.csv` | 4 KB | Per-solver mean certificate metrics (12 rows x 11 columns) |
| `results/pca_loadings.csv` | 4 KB | PCA loadings: PC1-PC5 by metric |
| `results/pca_pathology_axes.csv` | 4 KB | Per-solver PCA projections (12 x 5) |
| `results/pairwise_solver_distances.csv` | 4 KB | 12x12 Euclidean distance matrix |
| `results/clustering_summary.json` | 4 KB | Ward-linkage k=3 clustering result |

**Key finding**: PC1 (63.9%) = sigma/scale axis. Noisy solver nearest neighbor = linear_ridge (distance 3.34), NOT mean_baseline (distance 10.01). Aggregate metrics fail to separate degenerate solvers.

### C. `rsct-geocert/exp/s035_mast_reconcile/outputs/` -- Failure Taxonomy (4 files)

| File | Size | Description |
|------|------|-------------|
| `s018d_geocert_diagnosis.csv` | 4 KB | Per-solver GCF diagnosis: top1/top2/top3 modes with confidence |
| `s018d_geocert_diagnosis.json` | 8 KB | Same as CSV in JSON format |
| `s018d_failure_mode_counts.csv` | 1 KB | GCF-1.1: 9, GCF-1.2: 1, GCF-1.3: 1, GCF-2.1: 1 |
| `s018d_global_diagnosis.json` | 4 KB | Global signals (17 metrics) + 5 diagnosed modes at confidence=1.0 |

**Per-solver top1 failure mode**:

| Solver | Family | top1 | Key Signal |
|--------|--------|------|------------|
| mean_baseline | trivial | GCF-1.3 | Low R (0.29), low kappa (0.219) |
| noisy_solver | trivial | GCF-2.1 | Scalar Projection -- hidden by aggregation |
| linear_ridge | linear | GCF-1.2 | Label-Solver Coupling |
| svr_rbf | kernel | GCF-1.1 | Metric Compression (9 of 12 solvers) |
| knn | instance | GCF-1.1 | " |
| lightgbm | tree | GCF-1.1 | " |
| random_forest | tree | GCF-1.1 | " |
| mlp_regressor | neural | GCF-1.1 | " |
| hrm_regressor | hierarchical | GCF-1.1 | " |
| pca_v1 | spectral+tree | GCF-1.1 | " |
| spatial_lag_v1 | spatial | GCF-1.1 | " |
| gnn_v2 | graph | GCF-1.1 | " |

**Global diagnosis**: top1 = GCF-1.2 (Label-Solver Coupling), with GCF-2.1, GCF-2.2, GCF-2.3, GCF-3.1 all at confidence 1.0.

**Critical signals**: gate_entropy=0.0 (all EXECUTE), false_execute_risk=0.72, kappa_compression=0.61.

### D. `yrsn-experiments/exp/series_018/s018d_solver_diversity/` -- Source Code (8 files)

| File | Description |
|------|-------------|
| `sagemaker_s018d.py` | SageMaker launcher |
| `bootstrap.sh` | Container bootstrap (PYTHONIOENCODING fix) |
| `job_files/run_s018d_solver_diversity.py` | Main job script (28 KB, largest file) |
| `job_files/solvers.py` | 12-solver portfolio definition |
| `results/` | 4 parquet/json files (identical to rsct-geocert/data/s018d/) |

### E. `yrsn-experiments/exp/series_018/s018d_super_resolution/` -- Separate Experiment (3 files)

| File | Description |
|------|-------------|
| `DOE_LOCKED.md` | County-to-ZCTA super-resolution benchmark |
| `sagemaker_s018d.py` | SageMaker launcher |
| `job_files/run_s018d.py` | PCA(32)+GBDT county-to-ZCTA disaggregation |

### F. `yrsn/src/yrsn/ml/benchmarks/failure_taxonomy.py` -- Taxonomy Engine (1 file, 509 lines)

Defines the GeoCert Failure (GCF) taxonomy and diagnosis logic used by s035_mast_reconcile. The 9 modes:

| Category | Mode | Name |
|----------|------|------|
| GC1: Certificate Compression | GCF-1.1 | Metric Compression |
| | GCF-1.2 | Label-Solver Coupling |
| | GCF-1.3 | Gate Saturation |
| GC2: Measurement Blind Spots | GCF-2.1 | Scalar Projection |
| | GCF-2.2 | Gate Compression |
| | GCF-2.3 | Range Compression |
| GC3: Evaluation Confounds | GCF-3.1 | Target-Solver Conflation |
| | GCF-3.2 | Protocol Leakage |
| | GCF-3.3 | Benchmark Monoculture |

---

## 2. What Analyses Already Exist

| Analysis | Location | Status |
|----------|----------|--------|
| PCA on certificate metrics (5 PCs) | s018d_posthoc/results/ | DONE |
| Ward-linkage clustering (k=3) | s018d_posthoc/results/ | DONE |
| Pairwise solver distances | s018d_posthoc/results/ | DONE |
| GCF failure taxonomy diagnosis (per-solver) | s035_mast_reconcile/outputs/ | DONE |
| GCF global diagnosis (17 signals) | s035_mast_reconcile/outputs/ | DONE |
| Failure mode counts | s035_mast_reconcile/outputs/ | DONE |
| Super-resolution baseline (PCA+GBDT) | s018d_super_resolution/ | DONE |

---

## 3. What Does NOT Exist

| Analysis | Needed For | Notes |
|----------|-----------|-------|
| `failure_label` column in any parquet | Paper Table 1 | Must be joined from s018d_geocert_diagnosis.csv |
| Per-target kappa-threshold sweep | Injection-detection validation | analyze.py does PCA, not threshold analysis |
| alpha-rank-gap code as standalone | Paper claim re: rank preservation | Signal exists in global_diagnosis.json (0.85) but no reusable function |
| Injection-detection (stress_geocert.py) | NeurIPS Section 5 | failure_taxonomy.py diagnoses, but no standalone "inject noise, measure detection" script |
| Per-target certificate resolution | Beyond S018D scope | S018D has solver-level aggregates; per-target certificates are in certificate_rsn.parquet but not analyzed per-target in posthoc |
| Cross-protocol comparison | Paper Section 4 | Only imputation protocol used in S018D; no extrapolation/superres comparison |

---

## 4. Data Flow

```
yrsn-experiments/s018d_solver_diversity/
  run_s018d_solver_diversity.py + solvers.py
    -> SageMaker job
    -> results/*.parquet + summary.json
        |
        v (copied verbatim)
rsct-geocert/data/s018d/
  certificate_rsn.parquet (324 rows: 12 solvers x 27 targets)
  solver_metrics.parquet (12 rows: per-solver aggregates)
        |
        v (read by)
rsct-geocert/exp/s018d_posthoc/analyze.py
    -> PCA, clustering, distances
    -> results/ (7 files)
        |
        v (read by)
yrsn/src/yrsn/ml/benchmarks/failure_taxonomy.py
    + rsct-geocert/exp/s035_mast_reconcile/run_s035.py::classify_s018d()
    -> outputs/s018d_*.csv/.json (4 files)
```

---

## 5. Key Numbers

| Metric | Value |
|--------|-------|
| Solvers | 12 (2 trivial controls + 10 serious) |
| Targets | 27 |
| Certificates | 324 |
| PCA components to 100% variance | 5 |
| PC1 variance explained | 63.9% |
| GCF-1.1 (Metric Compression) count | 9 / 12 solvers |
| Gate entropy | 0.0 (all EXECUTE -- zero discrimination) |
| False execute risk | 0.72 |
| Noisy solver nearest neighbor | linear_ridge (3.34) not mean_baseline (10.01) |
| Alpha rank gap | 0.85 |

---

## 6. Implications for NeurIPS Plan

**What the user's plan assumed vs. what exists:**

| Plan Step | Assumed | Reality |
|-----------|---------|---------|
| Step 4: Build stress_geocert.py | Needs to be built | `failure_taxonomy.py` + `classify_s018d()` already do diagnosis; missing: standalone inject-and-detect script |
| Step 5: Injection-detection validation | Needs to be built | Global signals computed (17 metrics), but no formal injection protocol exists |
| Step 6: Re-diagnose certificates | Needs to be built | Already done in `s018d_geocert_diagnosis.csv` (12 solvers diagnosed) |
| Step 7: Aggregate diagnosis stats | Needs to be built | Already done: `s018d_failure_mode_counts.csv` + `s018d_global_diagnosis.json` |

**Bottom line**: Steps 6-7 are already complete. Step 4 is partially complete (taxonomy + diagnosis exist; injection script does not). Step 5 is the genuine gap.
