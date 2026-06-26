# DOE: Cross-Analysis Battery — Representation Orthogonality, Spatial Mechanism, Transfer, and Feature Stability

**Version:** 1.0
**Date:** 2026-06-26
**Parent experiment:** s035-model-ladder
**Status:** DESIGNED
**Depends on:** R0, R1, R2, R3, prithvi_embeddings, s037-TJEPA, spatial_sidecar_lisa, event_distance_matrix, money_table, certificates_r0/r1/r2

## Motivation

The s035 model ladder has 29 completed phases producing a rich artifact set across 5 flood scenarios, 3 representation levels (R0/R1/R2), TJEPA self-supervised embeddings (s037), and Prithvi satellite embeddings. Several high-value analytical questions can be answered from existing artifacts without new data collection or model training. These analyses address questions that peer reviewers will ask and that strengthen the paper's mechanistic claims.

## Analyses

### A1: Prithvi vs Tabular Orthogonality

**Question:** Do Prithvi-EO-2.0 satellite embeddings carry information independent of R0 tabular features, or are they redundant with flood zones and elevation?

**Hypothesis (AC-A1-1):** The canonical correlation between Prithvi embeddings and R0 features is < 0.70 (first canonical variate), indicating substantial independent information.

**Hypothesis (AC-A1-2):** Mutual information between Prithvi PC1-5 and top-5 R0 features (by importance) is < 0.30 nats.

**Method:**
1. For each scenario, load `{scenario}_prithvi_embeddings.parquet` (filter `source == "hls"` only, drop NaN fallback rows) and `{prefix}_event_features.parquet`.
2. Join on `zcta_id`. Deduplicate event_features to unique ZCTAs (take first event per ZCTA).
3. PCA-reduce Prithvi embeddings to 20 components (capturing ~90% variance).
4. Compute Canonical Correlation Analysis (CCA) between Prithvi PCA components and all numeric R0 features.
5. Compute pairwise mutual information (k-NN estimator, k=5) between Prithvi PC1-5 and top-5 R0 features by HistGBDT importance.
6. Report: first 5 canonical correlations, MI matrix, and whether Prithvi carries orthogonal signal.

**Inputs:**
- `results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet`
- `processed/{scenario}/{prefix}_event_features.parquet`

**Output:**
- `results/s035/cross_analysis/a1_prithvi_orthogonality.json`

**Quality gates:**
- Join yields >= 80% of HLS-covered ZCTAs (no silent drops)
- PCA components explain >= 80% cumulative variance before CCA

**Paper implication:** If orthogonal (AC-A1-1 PASS), Prithvi embeddings are a viable R4.5 representation arm. If redundant, satellite imagery adds no value beyond tabular flood zone features at ZCTA grain — itself a noteworthy negative result.

---

### A2: Kappa Reconstruct at R1/R2

**Question:** Does the adversarial kappa reconstruction test become discriminative at R1/R2 where spatial-lag features exist?

**Background:** The kappa_reconstruct phase ran at R0 and returned kappa_reconstruct=1.0 across all scenarios. This was expected — R0 has no spatial-lag features (`lag_cols=[]`), so W-matrix corruption has zero effect. The EXPERIMENT_STATUS.yaml notes: "Gate 3B promotion test needs R1/R2."

**Hypothesis (AC-A2-1):** At R1 (which includes 8 W-matrix spatial-lag features), W-matrix corruption reduces kappa_reconstruct below 0.95 in at least 3/5 scenarios.

**Hypothesis (AC-A2-2):** The magnitude of kappa degradation at R1 correlates with the scenario's Moran's I (from LISA sidecar) — scenarios with stronger spatial autocorrelation should show more sensitivity to W-corruption.

**Method:**
1. Reuse the `adversarial_reconstruct.py` job script with `--level r1` and `--level r2` flags.
2. For each scenario at each level:
   a. Train the standard HistGBDT model on the assembled features.
   b. Corrupt the W-matrix (row-permute adjacency, breaking spatial structure).
   c. Retrain on corrupted W-lag features.
   d. Compute kappa_reconstruct = R2_corrupted / R2_original.
3. Compare kappa_reconstruct across R0/R1/R2 levels.
4. Correlate kappa degradation at R1 with Moran's I from `lisa_results.json`.

**Inputs:**
- `processed/{scenario}/{prefix}_event_features.parquet`
- `raw/geocertdb2026/zcta_adjacency.parquet`
- `results/s035/sidecar/lisa_results.json`
- `results/s035/adversarial_reconstruct_{scenario}.json` (R0 baseline)

**Output:**
- `results/s035/cross_analysis/a2_kappa_reconstruct_r1r2.json`

**Quality gates:**
- R0 kappa_reconstruct must equal 1.0 (reproduces existing result as sanity check)
- R1 model must use >= 8 lag_cols (verify spatial-lag features present)

**Paper implication:** If R1 kappa degrades with W-corruption, Gate 3B becomes an active discriminator (not vacuously true). If kappa_reconstruct remains 1.0 at R1, the W-matrix features are truly not load-bearing even for adversarial reconstruction — stronger than the existing "not load-bearing for RMSE" verdict.

---

### A3: Cross-Scenario Transfer Matrix

**Question:** Which scenarios transfer to which? Does training on Houston predict Southwest Florida better than Riverside?

**Background:** The `event_distance_matrix` has Jaccard/Wasserstein distances between scenarios. R0-R2 predictions exist per scenario. But no cross-scenario prediction has been tested — each model was trained and evaluated within its own scenario.

**Hypothesis (AC-A3-1):** Leave-one-scenario-out transfer R2 correlates negatively with event distance (Wasserstein) from the event_distance_matrix — nearby scenarios in feature space transfer better.

**Hypothesis (AC-A3-2):** Houston is the best transfer source (largest n, most events), achieving R2 > 0.15 on at least 3 other scenarios.

**Method:**
1. For each source scenario S:
   a. Train R0 HistGBDT on all S data (all events, all folds combined).
   b. Predict on each target scenario T (T != S).
   c. Compute R2, RMSE on target.
2. Build 5x5 transfer matrix (source x target).
3. Correlate off-diagonal R2 with event_distance_matrix Wasserstein distances.
4. Report: transfer matrix, Spearman correlation, best/worst transfer pairs.

**Inputs:**
- `processed/{scenario}/{prefix}_event_features.parquet` (all 5 scenarios)
- `results/s035/sidecar/event_distance_matrix.json`

**Output:**
- `results/s035/cross_analysis/a3_transfer_matrix.json`

**Quality gates:**
- Feature alignment: all 5 scenarios must share the same R0 feature set (intersect columns)
- Within-scenario R2 must reproduce R0 results (sanity check)

**Paper implication:** Transfer matrix quantifies geographic generalizability. If transfer correlates with event distance, the event_distance_matrix becomes a predictive tool for deployment readiness. If Houston transfers well everywhere, it validates using Houston as the primary training scenario.

---

### A4: Feature Importance Stability Across Scenarios

**Question:** Do the same features matter across all 5 scenarios, or does each metro have a distinct driver profile?

**Hypothesis (AC-A4-1):** The rank correlation (Kendall's tau) of feature importances between any two scenarios is > 0.50 — a shared feature structure exists.

**Hypothesis (AC-A4-2):** The top-3 features by importance are the same in at least 4/5 scenarios.

**Method:**
1. For each scenario, load R0 result JSON.
2. Extract per-fold feature importances from HistGBDT (these are stored in fold results).
3. Average importances across folds to get a stable per-scenario ranking.
4. Compute pairwise Kendall's tau between all 5 scenario importance vectors.
5. Identify: top-3 per scenario, universally important features, scenario-specific features.
6. Repeat at R1 and R2 to see if importance stability changes with representation level.

**Inputs:**
- `results/s035/r0_{scenario}.json` (all 5)
- `results/s035/r1_hydrology_{scenario}.json` (all 5)
- `results/s035/r2_{scenario}.json` (all 5)

**Output:**
- `results/s035/cross_analysis/a4_feature_importance_stability.json`

**Quality gates:**
- Feature importances must be present in result JSONs (verify schema before computing)
- At least 20 features per scenario at R0

**Paper implication:** If stable (AC-A4-1 PASS), the model ladder findings generalize across metros. If unstable, the paper must qualify findings as metro-specific. Either way, a feature importance heatmap (5 scenarios x N features) is a strong paper figure.

---

### A5: Prithvi-TJEPA Representation Fusion Probe

**Question:** Are Prithvi (satellite) and TJEPA (tabular self-supervised) embeddings complementary or redundant?

**Background:** Both produce 1024-dim representations for the same ZCTAs. If orthogonal, a concatenated representation arm could be the actual R4.5 story.

**Hypothesis (AC-A5-1):** Cosine similarity between Prithvi and TJEPA embeddings for the same ZCTA is < 0.50 (they encode different information).

**Hypothesis (AC-A5-2):** PCA of the concatenated 2048-dim space retains more than 60% of variance in the first 20 components — the joint space is not degenerate.

**Method:**
1. For each scenario, load Prithvi embeddings (filter HLS-only) and TJEPA predictions parquet.
2. Join on zcta_id. Note: TJEPA results may have different column naming — extract TJEPA latent representations from the predictions parquet if available, or use the prediction residuals as a proxy.
3. Compute pairwise cosine similarity between Prithvi and TJEPA vectors for matched ZCTAs.
4. Concatenate to 2048-dim, run PCA, report variance explained.
5. If TJEPA latent embeddings are not directly available in the predictions parquet, this analysis is BLOCKED and should be noted.

**Inputs:**
- `results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet`
- `results/s037/s037_{scenario}_predictions.parquet`
- `results/s037/s037_{scenario}.json` (to check if embeddings are stored)

**Output:**
- `results/s035/cross_analysis/a5_prithvi_tjepa_fusion.json`

**Quality gates:**
- ZCTA join coverage >= 70% (Prithvi HLS-only x TJEPA coverage)
- Both embedding sets normalized before cosine computation

**Paper implication:** If complementary, this motivates a multi-modal representation arm combining foundation model satellite embeddings with self-supervised tabular representations — a novel contribution. If redundant, TJEPA already captures what satellite imagery provides.

---

### A6: Coverage Gap Geographic Overlap

**Question:** Do data coverage gaps (Prithvi fallback, hydrology missingness, FAST applicability) overlap geographically, creating systematic blind spots?

**Hypothesis (AC-A6-1):** ZCTAs missing Prithvi HLS data overlap with ZCTAs missing hydrology features at a rate > 2x random chance.

**Method:**
1. For each scenario, identify:
   a. Prithvi fallback ZCTAs (source == "fallback_no_data")
   b. Hydrology missing ZCTAs (from hydrology extraction metadata)
   c. FAST-excluded scenarios (riverside_coachella, new_orleans by design)
2. Compute overlap: Jaccard index between Prithvi-missing and hydrology-missing ZCTA sets.
3. Map geographic pattern: are gaps coastal, rural, or correlated with population density?
4. Report: per-scenario gap sets, overlap statistics, geographic characterization.

**Inputs:**
- `results/s035/prithvi_embeddings/{scenario}_prithvi_meta.json`
- `results/s035/hydrology_extraction_{scenario}.json`
- `processed/{scenario}/{prefix}_event_features.parquet` (for lat/lon)

**Output:**
- `results/s035/cross_analysis/a6_coverage_gap_overlap.json`

**Quality gates:**
- Both Prithvi and hydrology metadata must report per-ZCTA coverage status

**Paper implication:** If gaps overlap, the paper must disclose a systematic coverage limitation. If independent, missing data is random and less concerning for generalizability claims.

---

## Execution Plan

**Priority order** (by paper impact):

| Priority | Analysis | Complexity | New code needed? |
|----------|----------|-----------|-----------------|
| 1 | A3 Transfer matrix | Low | New job script |
| 2 | A4 Feature importance stability | Low | New job script |
| 3 | A1 Prithvi orthogonality | Medium | New job script |
| 4 | A2 Kappa reconstruct R1/R2 | Medium | Extend existing adversarial_reconstruct.py |
| 5 | A6 Coverage gap overlap | Low | New job script |
| 6 | A5 Prithvi-TJEPA fusion | Medium | New job script (may be BLOCKED) |

A3 and A4 are highest priority because reviewers will ask "does this generalize?" and "what drives the model?" A1 and A2 answer mechanistic questions. A6 is a disclosure requirement. A5 is exploratory.

## Compute Requirements

All analyses run on existing S3 artifacts. No new data collection. No new model training for A1/A3/A4/A5/A6 (join + compute on downloaded parquets). A2 requires retraining HistGBDT at R1/R2 with corrupted W-matrix.

Estimated per job: ml.m5.xlarge, < 30 min, < 10 GB volume.

## Falsification Criteria

Each analysis has explicit PASS/FAIL thresholds in hypotheses. Negative results are equally publishable:
- A1 FAIL (Prithvi redundant) → satellite imagery adds nothing beyond tabular at ZCTA grain
- A2 FAIL (kappa still 1.0 at R1) → W-matrix truly irrelevant even for adversarial reconstruction
- A3 FAIL (no transfer correlation) → event distance metric is not predictive of transferability
- A4 FAIL (importance unstable) → findings are metro-specific, not generalizable
- A5 FAIL (representations redundant) → TJEPA already captures satellite signal
- A6 FAIL (gaps don't overlap) → missing data is random, not systematic
