# Experiment Evidence

Pre-computed results from all experiments cited in the paper.
Each subdirectory contains raw outputs and analysis scripts so reviewers can
verify reported numbers without re-running the full pipeline.

## Experiments

### S019D — Theory Kappa Benchmark (Section 6, Table 2)

**Question:** Does the theory-derived compatibility score (kappa) predict
which representation will perform best on each task?

- **Design:** 6 representations x 27 targets x 5 folds x 3 seeds (42, 123, 456)
- **Key result:** Within-task Spearman rho = 0.947 [0.913, 0.973] (B=1000 bootstrap CI)
- **Files:**
  - `s019d/results/` — per-seed raw results (810 records each) and summaries
  - `s019d/results/seed_{42,123,456}/` — individual seed outputs
  - `s019d_bootstrap/` — bootstrap confidence intervals (CSV + JSON)

### S019A — Certificate Invariance Gradient (Section 6)

**Question:** How much of the variance in solver performance comes from
representation choice vs. solver choice vs. their interaction?

- **Design:** 3 representations x 3 solvers x 3 targets, two-way ANOVA
- **Key result:** Embedding eta-squared (0.55-0.69) dominates solver (0.14-0.22)
  and interaction (0.04-0.09) by 6-16x
- **Files:**
  - `s019a/results/s019a_results.json` — raw ANOVA results
  - `s019a_posthoc/` — post-hoc analysis (Tukey HSD, contrasts, assumptions)
  - `s019a_posthoc/results/tab_s019a_*.csv` — tables cited in paper

### S018D-Posthoc — Certificate Structure Analysis (Appendix)

**Question:** Do aggregate certificate metrics capture solver pathologies?

- **Key result:** PCA reveals 3 orthogonal pathology axes but fails to
  distinguish degenerate solvers from serious ones
- **Files:**
  - `s018d_posthoc/results/` — PCA loadings, pairwise distances, clustering
  - `s018d_posthoc/analyze.py` — analysis script (re-runnable)

### Predictions — Solver Metrics and Certificates (Section 6, Table 2)

Pre-computed solver outputs used to generate paper tables and figures.

| File | Contents |
|------|----------|
| `predictions/solver_metrics.parquet` | Per-solver, per-task R-squared values |
| `predictions/target_family_metrics.parquet` | Metrics aggregated by target family |
| `predictions/certificate_rsn.parquet` | Issued R/S/N certificate triples |
| `predictions/solver_metric_summary.csv` | Summary statistics across solvers |
| `predictions/pairwise_solver_distances.csv` | Inter-solver distance matrix |
| `predictions/pca_loadings.csv` | PCA loadings on certificate space |

## Verified Numbers

`verified_numbers.md` cross-checks every key number in the paper against the
raw data files in this directory. Each claim is marked VERIFIED, CONSISTENT,
or UNVERIFIED with the source file and computation method.

## Reproducing from Raw Data

The experiment code lives in `code/experiments/series_019/`. To regenerate
results from the benchmark dataset:

1. Download `georsct_table.parquet` from [HuggingFace](https://huggingface.co/datasets/rudymartin/georsct)
2. Pre-computed representations are in `representations/*.npz` on HuggingFace
3. Run scripts in `code/experiments/series_019/s019d_comprehensive_theory_kappa/`
   and `code/experiments/series_019/s019a_certificate_invariance_gradient/`

See `code/experiments/series_019/` for DOE documents, checklists, and
SageMaker launcher configs used for the original runs.
