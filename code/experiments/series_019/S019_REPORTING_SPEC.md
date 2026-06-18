# S019 Reporting Specification

What every S019 experiment must produce for paper tables, appendices,
and post-hoc analysis scripts. This is the contract between run scripts
and reporting scripts.

---

## 1. Per-Cell Fields (from `certify_group` + run script enrichment)

Every row in `s019X_results.json` must contain:

### 1a. Identity

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `target` | str | run script | CONUS-27 task name (e.g. "diabetes") |
| `target_family` | str | run script | "health", "socioeconomic", "environmental" |
| `embedding` | str | certify_group | Embedding arm name (e.g. "pca_v1") |
| `solver` | str | run script | Solver name (e.g. "histgbdt") |
| `fold` | int | run script | CV fold index (0..4) |
| `seed` | int | run script | Random seed (42, 123, 456) |

### 1b. Performance

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `r2` | float | certify_group | R-squared on test fold |
| `ba_test` | float | certify_group | Balanced accuracy (classification) |
| `n_train` | int | certify_group | Training samples |
| `n_test` | int | certify_group | Test samples |
| `wall_clock_s` | float | run script | Fold wall-clock seconds |

### 1c. Simplex (R + S_sup + N = 1)

| Field | Type | Source | Paper symbol |
|-------|------|--------|--------------|
| `R` | float | certify_group | R |
| `S_sup` | float | certify_group | S_sup |
| `N` | float | certify_group | N |
| `alpha` | float | certify_group | alpha = R / (R + N) |
| `simplex_sum` | float | certify_group | Integrity check (~1.0) |

### 1d. Kappa (proxy and theory)

| Field | Type | Source | Paper symbol |
|-------|------|--------|--------------|
| `proxy_kappa` | float | certify_group | kappa_code = R*(1-N) |
| `proxy_kappa_mean` | float | certify_group | Mean per-sample proxy kappa |
| `proxy_kappa_std` | float | certify_group | Std of per-sample proxy kappa |
| `theory_kappa` | float | certify_group | kappa_theory = D*/D (LOO) |
| `theory_kappa_mean` | float | certify_group | Mean per-sample theory kappa |
| `theory_kappa_min` | float | certify_group | Min per-sample theory kappa |
| `theory_sigma` | float | certify_group | Std of per-sample theory kappa |

### 1e. Turbulence & Diagnostics

| Field | Type | Source | Paper symbol |
|-------|------|--------|--------------|
| `sigma` | float | certify_group | compute_sigma_request(N) |
| `omega` | float | certify_group | Distributional reliability |
| `entropy` | float | certify_group | Simplex entropy |
| `collapse_risk` | float | certify_group | Collapse risk indicator |

### 1f. Gate Evaluation (S019D only)

Each row contains `gate_flat` and `gate_oobleck` sub-dicts:

| Field | Type | Description |
|-------|------|-------------|
| `gate_decision` | str | "EnforcementDecision.EXECUTE" / REJECT / etc. |
| `kappa_req` | float | Required kappa threshold (Gate 3) |
| `margin` | float | theory_kappa - kappa_req |
| `gate_reached` | str | Last gate reached |
| `failure_reason` | str | None if passed, reason if failed |

### 1g. Derived (computed by run script)

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `task_residual_floor` | float | run script | 1 - best_embedding_r2 per task |

---

## 2. Paper Tables Requiring S019 Data

### Table 2: Main Results (Section 5)

Grouped by target_family {health, socioeconomic, environmental}:
- Mean R2 by embedding
- Mean theory_kappa by embedding
- Gate pass rate (flat vs oobleck)
- Cross-family spread

**Source fields**: `r2`, `theory_kappa`, `gate_flat.gate_decision`, `gate_oobleck.gate_decision`, `target_family`, `embedding`

### Appendix F: Experiment Ledger

Per-experiment row:
- Experiment ID (S019A, S019D)
- Primary metric + value
- Instance type
- Wall-clock minutes
- Verdict (PASS/FAIL with threshold)

**Source fields**: `wall_clock_s` (sum), per-experiment summary

### S019A-Specific Hypotheses (H1-H7)

| ID | Hypothesis | Required Fields |
|----|-----------|-----------------|
| H1 | Simplex integrity | `simplex_sum` (max deviation < 0.01) |
| H2 | alpha range | `alpha` per embedding (spread > 0.02) |
| H3 | kappa range | `proxy_kappa` or `theory_kappa` per embedding (spread > 0.02) |
| H4 | sigma stability | `sigma` (< 0.30 for >= 20 tasks) |
| H5 | Cross-seed consensus | `theory_kappa` across seeds (Pearson r > 0.90) |
| H6 | Gate discrimination | `gate_flat.gate_decision` (> 1 distinct decision) |
| H7 | kappa-residual correlation | `theory_kappa` + `r2` (Spearman rho(kappa, |r|) < -0.10) |

### S019D-Specific Tables

| Table | Content | Required Fields |
|-------|---------|-----------------|
| Embedding comparison (6-way) | R2, kappa, gate rate per embedding x family | `r2`, `theory_kappa`, `gate_*`, `embedding`, `target_family` |
| Proxy vs theory correlation | Spearman rho, scatter | `proxy_kappa`, `theory_kappa` |
| Oobleck flip analysis | Certificates that change decision | `gate_flat.gate_decision`, `gate_oobleck.gate_decision` |
| TRF by task | Best-embedding R2 as noise floor | `task_residual_floor`, `target` |
| Cross-seed consensus (FC-5) | Per-task kappa vectors across seeds | `theory_kappa`, `seed`, `target`, `embedding` |

---

## 3. Summary Files

Each experiment's run script must produce:

| File | Format | Content |
|------|--------|---------|
| `s019X_results.json` | JSON array | All per-cell rows (Section 1 schema) |
| `s019X_summary.json` | JSON object | Aggregate statistics, per-task summaries, timing |
| `s019X_task_residual_floor.json` | JSON object | Per-task TRF estimates (S019D) |

---

## 4. Post-Hoc Analysis Outputs

Each `analyze_s019X.py` script must produce:

### 4a. CSV Tables (for LaTeX ingestion)

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `r2_by_embedding.csv` | targets x embeddings | mean, std, median R2 | Table 2 data |
| `kappa_by_embedding.csv` | targets x embeddings | theory_kappa, proxy_kappa | Kappa comparison |
| `gate_results.csv` | targets x embeddings x gatekeeper | decision, kappa_req, margin | Gate analysis |
| `hypotheses.csv` | H1-H7 | metric, threshold, observed, verdict | Hypothesis table |
| `certificates.csv` | all cells | full simplex + kappa + gate | Master certificate table |
| `contrast_pairwise.csv` | embedding pairs | delta_r2, delta_kappa, p_value | Tukey/pairwise |

### 4b. Certificate Export

| File | Format | Purpose |
|------|--------|---------|
| `georsct/data/s019X/certs/*.json` | Per-cell JSON | Archival certificates |
| `georsct/data/s019X/certificates.parquet` | Parquet | Machine-readable full table |
| `georsct/data/s019X/summary.json` | JSON | Experiment summary |

### 4c. Cross-Seed Outputs (S019D with --all-seeds)

| File | Format | Purpose |
|------|--------|---------|
| `cross_seed_consensus.csv` | targets x seeds | Pearson r for FC-5 |
| `cross_seed_kappa_vectors.csv` | targets x embeddings x seeds | Raw kappa for scatter plots |

---

## 5. Field Gaps & Missing Data

### Currently Emitted by run_s019d.py: COMPLETE
All fields in Section 1 are present. No gaps.

### NOT Emitted (must be computed post-hoc)

| Statistic | Computed From | Where |
|-----------|---------------|-------|
| Tukey pairwise comparisons | `r2` grouped by embedding | analyze_s019d.py |
| ANOVA F-statistic | `r2` grouped by embedding | analyze_s019d.py |
| Cross-seed Pearson r | `theory_kappa` across seed files | analyze_s019d.py |
| Calibrated policy sensitivity | Re-evaluate gates with different thresholds | analyze_s019d.py |
| Per-sample kappa arrays | `_kappa_per_proxy`, `_kappa_per_theory` | Stripped by `_` prefix filter in run script |

### Per-Sample Data Decision

The `_kappa_per_proxy` and `_kappa_per_theory` arrays are stripped from
JSON output (they start with `_`). This is intentional -- they are large
(n_test floats per cell) and the aggregate statistics (mean, std, min)
capture what the paper needs. If per-sample scatter plots are needed,
the post-hoc script must re-run `certify_group` locally.

---

## 6. Experiment Matrix

| Experiment | Embeddings | Solvers | Tasks | Folds | Seeds | Gatekeepers | Total Fits |
|-----------|-----------|---------|-------|-------|-------|-------------|------------|
| S019A | 3 (pca, spatial_lag, gnn) | 1 (histgbdt) | 27 | 5 | 1 | flat + oobleck | 405 |
| S019C | 3 | 1 | 3 (selected) | 5 | 1 | oobleck (active) | 45 |
| S019D | 6 (pca, spatial_lag, gnn, geo, noisy, domain) | 1 | 27 | 5 | 3 (42,123,456) | flat + oobleck | 2430 |

---

## 7. Reporting Script Checklist

Before marking an experiment as "paper-ready":

- [ ] All cells completed (no failed targets)
- [ ] H1 simplex integrity passes (max dev < 0.01)
- [ ] Noisy control rejected by both gatekeepers (S019D sanity check)
- [ ] Cross-seed consensus > 0.90 (FC-5, S019D only)
- [ ] Certificate JSONs exported to georsct/data/
- [ ] CSV tables generated for LaTeX
- [ ] Wall-clock time recorded for Appendix H
- [ ] Git hash of run code recorded in summary
