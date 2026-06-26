# DOE: Cross-Analysis Battery — Representation Orthogonality, Spatial Mechanism, Transfer, and Feature Stability

**Version:** 2.0
**Date:** 2026-06-26
**Parent experiment:** s035-model-ladder
**Status:** DESIGNED
**Depends on:** R0, R1, R2, R3, prithvi_embeddings, s037-TJEPA, spatial_sidecar_lisa, event_distance_matrix, money_table, certificates_r0/r1/r2

**MMAR review:** v1.0 reviewed 2026-06-26 by claude, deepseek, deepseek-r1, kimi, nemotron.
2 critical, 6 serious findings. All addressed in v2.0. See `mmar-output/doe_cross_analysis/findings.md`.

## Motivation

The s035 model ladder has 29 completed phases producing a rich artifact set across 5 flood scenarios, 3 representation levels (R0/R1/R2), TJEPA self-supervised embeddings (s037), and Prithvi satellite embeddings. Several high-value analytical questions can be answered from existing artifacts without new data collection. These analyses address questions that peer reviewers will ask and that strengthen the paper's mechanistic claims.

## Multiple Comparison Correction

This DOE contains 6 analyses with a total of 10 primary hypotheses. All hypothesis tests
use Holm-Bonferroni correction with family-wise alpha = 0.05 across the 10 tests. Each
analysis reports both raw and corrected p-values. Descriptive analyses (importance heatmaps,
coverage maps) are not subject to correction.

## Analyses

### A1: Prithvi vs Tabular — Predictive Utility Test

**Question:** Do Prithvi-EO-2.0 satellite embeddings improve downstream flood damage prediction when added to R0 tabular features?

**Background:** Prithvi embeddings (1024-dim mean-pooled patch tokens) capture land-cover variation from HLS satellite imagery. The question is not whether they are statistically independent of R0 (low CCA could mean independent noise), but whether they carry predictive signal beyond what R0 already provides.

**Hypothesis (AC-A1-1):** Adding Prithvi PCA features (top-k components explaining >= 90% variance) to R0 features improves held-out R2 by >= 0.02 in at least 3/5 scenarios (paired t-test across folds, Holm-corrected alpha).

**Hypothesis (AC-A1-2):** The canonical correlation between Prithvi PCA and R0 features is reported descriptively (no PASS/FAIL threshold) to characterize the degree of linear redundancy.

**Method:**
1. For each scenario, load `{scenario}_prithvi_embeddings.parquet` (filter `source == "hls"` only, drop NaN fallback rows) and `{prefix}_event_features.parquet`.
2. Join on `zcta_id`. Retain event-level grain (do NOT deduplicate to unique ZCTAs). Each event row gets the same Prithvi embedding for its ZCTA — this is correct because Prithvi captures static land cover, not event dynamics. The join is many-to-one (events to ZCTA embedding). Log the join: n_events_before, n_events_after, n_zctas_matched.
3. PCA-reduce Prithvi embeddings to k components explaining >= 90% cumulative variance (computed on unique ZCTAs to avoid inflating variance from duplicated rows).
4. Train HistGBDT with 5-fold spatial-blocked CV on two arms:
   - **R0-only**: original R0 features (baseline, must reproduce existing R0 R2 within +/- 0.01).
   - **R0+Prithvi**: R0 features + Prithvi PCA components.
5. Compute per-fold R2 delta. Report paired t-test (R0+Prithvi vs R0-only) with effect size (Cohen's d).
6. Descriptive: compute CCA between Prithvi PCA and R0 numeric features on unique ZCTAs. Report first 5 canonical correlations without a PASS/FAIL threshold.

**Inputs:**
- `results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet`
- `processed/{scenario}/{prefix}_event_features.parquet`

**Output:**
- `results/s035/cross_analysis/a1_prithvi_utility.json`

**Quality gates:**
- Event-level join yields >= 80% of event rows (no silent drops from missing ZCTA matches)
- PCA computed on unique ZCTAs, not duplicated event rows
- R0-only arm reproduces existing R0 R2 within +/- 0.01 (sanity check)
- n_events per scenario >= 100 for statistical power

**Paper implication:** If R2 improves (AC-A1-1 PASS), Prithvi embeddings are a viable R4.5 representation arm with measured effect size. If no improvement, satellite imagery adds no predictive value beyond tabular at ZCTA grain — a noteworthy negative result that bounds the value of foundation model embeddings for this task.

---

### A2: Kappa Reconstruct at R1/R2

**Question:** Does the adversarial kappa reconstruction test become discriminative at R1/R2 where spatial-lag features exist?

**Background:** The kappa_reconstruct phase ran at R0 and returned kappa_reconstruct=1.0 across all scenarios. This was expected — R0 has no spatial-lag features (`lag_cols=[]`), so W-matrix corruption has zero effect. The EXPERIMENT_STATUS.yaml notes: "Gate 3B promotion test needs R1/R2."

**Hypothesis (AC-A2-1):** At R1 (which includes 8 W-matrix spatial-lag features), W-matrix corruption reduces kappa_reconstruct below 0.95 in at least 3/5 scenarios.

**Method:**
1. Reuse the `adversarial_reconstruct.py` job script with `--level r1` and `--level r2` flags.
2. For each scenario at each level:
   a. Train the standard HistGBDT model using spatial-blocked 5-fold CV (same fold structure as original s035 runs).
   b. Corrupt the W-matrix (row-permute adjacency, breaking spatial structure).
   c. Retrain on corrupted W-lag features using the same fold structure.
   d. Compute kappa_reconstruct = mean(R2_corrupted_per_fold) / mean(R2_original_per_fold), where R2 is computed on held-out test folds only.
3. Compare kappa_reconstruct across R0/R1/R2 levels.
4. Report per-scenario, per-level kappa_reconstruct with 95% CI from fold variance.

**Note on AC-A2-2 (removed):** v1.0 proposed correlating kappa degradation with Moran's I across scenarios. MMAR review correctly identified that n=5 scenarios provides negligible statistical power for a correlation test, and there is no established mechanistic link between outcome autocorrelation and feature reconstruction sensitivity. Removed.

**Inputs:**
- `processed/{scenario}/{prefix}_event_features.parquet`
- `raw/geocertdb2026/zcta_adjacency.parquet`
- `results/s035/adversarial_reconstruct_{scenario}.json` (R0 baseline for sanity check)

**Output:**
- `results/s035/cross_analysis/a2_kappa_reconstruct_r1r2.json`

**Quality gates:**
- R0 kappa_reconstruct must equal 1.0 (reproduces existing result as sanity check)
- R1 model must use >= 8 lag_cols (verify spatial-lag features present)
- R2 is computed on held-out folds only (never on training data)
- Same fold structure as original s035 runs (reproducibility)

**Paper implication:** If R1 kappa degrades with W-corruption, Gate 3B becomes an active discriminator (not vacuously true). If kappa_reconstruct remains 1.0 at R1, the W-matrix features are truly not load-bearing even for adversarial reconstruction — stronger than the existing "not load-bearing for RMSE" verdict.

---

### A3: Cross-Scenario Transfer Matrix

**Question:** Which scenarios transfer to which? Does training on Houston predict Southwest Florida better than Riverside?

**Background:** The `event_distance_matrix` has Jaccard/Wasserstein distances between scenarios. R0-R2 predictions exist per scenario. But no cross-scenario prediction has been tested — each model was trained and evaluated within its own scenario.

**Hypothesis (AC-A3-1):** Leave-one-scenario-out transfer R2 correlates negatively with event distance (Wasserstein) from the event_distance_matrix — nearby scenarios in feature space transfer better. Partial correlation controlling for source scenario sample size is used to address the confound that larger training sets produce better models regardless of distance.

**Hypothesis (AC-A3-2):** At least one source scenario achieves transfer R2 > 0.10 on at least 2 target scenarios (positive transfer exists).

**Method:**
1. Compute feature intersection across all 5 scenarios. Report any features present in some but not all scenarios. Use only the shared feature set for transfer.
2. For each source scenario S:
   a. Train R0 HistGBDT on all S data (all events combined) — this is a transfer-specific model, not comparable to within-scenario CV results.
   b. Predict on each target scenario T (T != S) using the shared feature set.
   c. Compute R2, RMSE on target.
3. Build 5x5 transfer matrix (source x target). Diagonal is left empty (not comparable to CV R2).
4. Baselines:
   - **Mean-prediction baseline**: predict target mean for all rows. Transfer R2 must exceed this.
   - **Random-feature baseline**: train on S with shuffled features, predict on T. Bounds chance-level transfer.
5. Compute partial Spearman correlation between off-diagonal R2 and Wasserstein distance, controlling for source scenario n_events.
6. Report per-feature distribution comparison (KS test) between each source-target pair to characterize distribution shift.

**Inputs:**
- `processed/{scenario}/{prefix}_event_features.parquet` (all 5 scenarios)
- `results/s035/sidecar/event_distance_matrix.json`

**Output:**
- `results/s035/cross_analysis/a3_transfer_matrix.json`

**Quality gates:**
- Feature intersection yields >= 25 shared features (sufficient for meaningful model)
- Each source scenario has >= 50 training rows
- Transfer R2 compared against mean-prediction baseline (not against within-scenario CV R2)

**Paper implication:** Transfer matrix quantifies geographic generalizability. If transfer correlates with event distance (controlling for n), the event_distance_matrix becomes a predictive tool for deployment readiness. If no scenario transfers positively, the paper must scope findings as metro-specific.

---

### A4: Feature Importance Stability Across Scenarios

**Question:** Do the same features matter across all 5 scenarios, or does each metro have a distinct driver profile?

**Background:** R0 result JSONs store per-run metrics but do not serialize HistGBDT feature importances. This analysis requires retraining to extract importances.

**Hypothesis (AC-A4-1):** The rank correlation (Kendall's tau) of feature importances between any two scenarios is > 0.40 — a shared feature structure exists. Threshold justification: tau > 0.40 is conventionally "moderate agreement" (Kendall 1938); below this, rankings are effectively independent. Holm-corrected across all 10 pairwise comparisons (5 choose 2).

**Hypothesis (AC-A4-2):** The top-3 features by importance are the same in at least 3/5 scenarios (majority agreement).

**Method:**
1. For each scenario, load event_features parquet and retrain R0 HistGBDT with 5-fold spatial-blocked CV (same fold structure as original s035).
2. Extract HistGBDT `feature_importances_` (split-based importance) from each fold's fitted model.
3. Average importances across folds to get a stable per-scenario importance vector.
4. Compute all 10 pairwise Kendall's tau between 5 scenario importance vectors.
5. Identify: top-5 per scenario, universally important features (top-5 in >= 4 scenarios), scenario-specific features (top-5 in exactly 1 scenario).
6. Repeat at R1 and R2 to see if importance stability changes with representation level.
7. Report importance heatmap data: 5 scenarios x N features, normalized to sum=1 per scenario.

**Inputs:**
- `processed/{scenario}/{prefix}_event_features.parquet` (all 5 scenarios)

**Output:**
- `results/s035/cross_analysis/a4_feature_importance_stability.json`

**Quality gates:**
- Retrained R0 R2 matches original R0 R2 within +/- 0.01 per scenario (verifies identical model setup)
- At least 25 features per scenario at R0
- Feature names identical across scenarios (same R0 feature set)

**Paper implication:** If stable (AC-A4-1 PASS), the model ladder findings generalize across metros — the same flood risk drivers operate everywhere. If unstable, the paper must qualify findings as metro-specific and the importance heatmap becomes a key figure showing geographic heterogeneity.

---

### A5: Prithvi-TJEPA Representation Fusion Probe

**Status: BLOCKED**

**Reason:** s037 TJEPA predictions parquet contains only `y_true`/`y_pred` columns — no latent embeddings are serialized. The TJEPA encoder produces internal representations during training but does not export them. Computing cosine similarity or PCA between Prithvi 1024-dim embeddings and 1-dim prediction residuals is type-invalid (MMAR critical finding).

**Unblocking requirement:** Re-run s037 with a modified job script that serializes the TJEPA encoder's latent representation (the output of the masked autoencoder's encoder, before the prediction head) to a separate `{scenario}_tjepa_embeddings.parquet`. This requires:
1. Modifying the TJEPA training loop to export `encoder(x)` for each ZCTA after training.
2. Re-running all 5 scenarios.
3. Approximately 1 hour of SageMaker compute.

**If unblocked, the analysis would test:**
- (AC-A5-1) Cosine similarity between Prithvi and TJEPA embeddings for matched ZCTAs < 0.50.
- (AC-A5-2) PCA of concatenated 2048-dim space retains > 60% variance in first 20 components.
- Random embedding baseline: cosine with 1024-dim random Gaussian vectors (expected ~0 in high-dim).

**Decision:** Defer until A1 results are known. If Prithvi adds no predictive value (A1 FAIL), fusion is moot.

---

### A6: Coverage Gap Geographic Overlap

**Question:** Do data coverage gaps (Prithvi fallback, hydrology missingness, FAST applicability) overlap geographically, creating systematic blind spots?

**Hypothesis (AC-A6-1):** The overlap between Prithvi-missing and hydrology-missing ZCTA sets is greater than expected by chance (Fisher's exact test, one-tailed, Holm-corrected alpha).

**Method:**
1. For each scenario, identify:
   a. Prithvi fallback ZCTAs (`source == "fallback_no_data"` in prithvi_meta.json `hls_coverage` list — absent ZCTAs are fallback)
   b. Hydrology missing ZCTAs (from hydrology extraction metadata, coverage < 100%)
   c. FAST-excluded scenarios (riverside_coachella, new_orleans by design — report but do not test)
2. For each scenario, build a 2x2 contingency table:
   |  | Hydro present | Hydro missing |
   |--|--------------|--------------|
   | Prithvi present | a | b |
   | Prithvi missing | c | d |
3. Fisher's exact test (one-tailed: overlap greater than chance). Report odds ratio and p-value.
4. Compute Jaccard index for descriptive comparison.
5. Descriptive geographic characterization: for ZCTAs in the overlap (missing both), report median latitude, longitude, and distance to coast. Compare to scenario-wide medians. No hypothesis test — purely descriptive.

**Inputs:**
- `results/s035/prithvi_embeddings/{scenario}_prithvi_meta.json`
- `results/s035/hydrology_extraction_{scenario}.json`
- `processed/{scenario}/{prefix}_event_features.parquet` (for lat/lon)

**Output:**
- `results/s035/cross_analysis/a6_coverage_gap_overlap.json`

**Quality gates:**
- Both Prithvi and hydrology metadata must report per-ZCTA coverage status
- Fisher's test requires >= 5 expected count in each cell (report exact test regardless, but flag if violated)

**Paper implication:** If gaps overlap significantly, the paper must disclose a systematic coverage limitation and characterize its geographic pattern. If independent, missing data is random and less concerning for generalizability claims.

---

## Execution Plan

**Priority order** (by paper impact and feasibility):

| Priority | Analysis | Complexity | New code needed? | Blocked? |
|----------|----------|-----------|-----------------|----------|
| 1 | A3 Transfer matrix | Medium | New job script | No |
| 2 | A4 Feature importance | Medium | New job script (retrain) | No |
| 3 | A1 Prithvi utility | Medium | New job script (retrain) | No |
| 4 | A2 Kappa reconstruct R1/R2 | Medium | Extend adversarial_reconstruct.py | No |
| 5 | A6 Coverage gap overlap | Low | New job script | No |
| 6 | A5 Prithvi-TJEPA fusion | Medium | Modify s037 + new script | **BLOCKED** |

**Rationale:** A3 answers "does this generalize?" — the first reviewer question. A4 answers "what drives the model?" A1 answers "do satellite embeddings help?" A2 completes the Gate 3B story. A6 is a disclosure requirement. A5 is deferred pending A1 results and TJEPA embedding export.

## Compute Requirements

A1, A3, A4 require retraining HistGBDT models (lightweight — < 10 min per scenario on ml.m5.xlarge). A2 extends existing adversarial_reconstruct.py. A6 is pure join + contingency table. No GPU required.

Estimated per job: ml.m5.xlarge (4 vCPU, 16 GB), < 30 min, 10 GB volume, PyTorch-CPU image.

## Falsification Criteria

Each analysis has explicit PASS/FAIL thresholds with stated justification. Negative results are equally publishable:
- A1 FAIL (Prithvi no uplift) → satellite imagery adds no predictive value beyond tabular at ZCTA grain
- A2 FAIL (kappa still 1.0 at R1) → W-matrix truly irrelevant even for adversarial reconstruction
- A3 FAIL (no transfer) → findings are metro-specific, not geographically generalizable
- A4 FAIL (importance unstable) → different metros have different flood risk drivers
- A5 BLOCKED → deferred pending TJEPA embedding export
- A6 FAIL (gaps independent) → missing data is random, not systematic

## MMAR v1.0 Findings Addressed

| Finding | Severity | Resolution |
|---------|----------|------------|
| A1 sample collapse (deduplicate to ZCTA) | Critical | Retain event-level grain; PCA on unique ZCTAs only |
| A5 residual proxy type-invalid | Critical | Hard-gated as BLOCKED; no fallback proxy |
| A1 thresholds arbitrary (CCA/MI) | Serious | Replaced with downstream prediction test (R2 delta) |
| A3 sanity check breaks CV | Serious | Diagonal left empty; no comparison to CV results |
| A2 Moran's I correlation n=5 | Serious | Removed AC-A2-2 entirely |
| A4 assumes per-fold importances | Serious | Verified schema lacks importances; retrain to extract |
| A2 kappa holdout not specified | Serious | Explicit: held-out test folds only |
| A3 sample size confound | Serious | Partial correlation controlling for n_events |
| Multiple comparison correction | Minor | Holm-Bonferroni across 10 hypotheses, family alpha=0.05 |
| k=5 MI unjustified | Minor | MI removed (replaced with prediction test) |
| Kendall's tau threshold arbitrary | Minor | Justified: tau>0.40 = conventional "moderate agreement" |
| A6 overlap criterion unanchored | Minor | Fisher's exact test with odds ratio |
| A3 missing baseline | Minor | Mean-prediction + random-feature baselines added |
| A5 missing random baseline | Minor | Added to A5 design (when unblocked) |
